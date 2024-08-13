from GNN.MolToGraph import create_pyg_data_lst
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from torch_geometric import loader

from GNN.GNNModel import GNNModel
from GNN.GraphGNNModel import GraphGNNModel
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- config -----------------
device = torch.device('mps') # mac版本是'mps'
# ------------------------------------------


# ----------------- STEP1 数据定义 -----------------
solv = pd.read_csv('data/FreeSolv.tsv', sep=';')
smiles = solv['SMILES'].values

train_idx, test_idx = train_test_split(range(len(smiles)), test_size=0.2, random_state=0)
smiles_train, smiles_test = smiles[train_idx], smiles[test_idx]

y = (solv['experimental value (kcal/mol)']<= solv['experimental value (kcal/mol)'].median()).astype(int)
y_train, y_test = y[train_idx], y[test_idx]

train_loader = loader.DataLoader(create_pyg_data_lst(smiles_train, y_train, device='mps'), batch_size=16)
test_loader = loader.DataLoader(create_pyg_data_lst(smiles_test, y_test, device='mps'), batch_size=16)

# ----------------- STEP2 模型定义 -----------------
model = GraphGNNModel(
    D_in=79,
    D_hidden=256,
    D_out=1,
    dp_rate_linear=0.5,
    dp_rate=0.0,
    num_layers=3,
    ).to(device)

# ----------------- STEP3 定义优化器-----------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# ----------------- STEP4 定义损失函数 -----------------
criterion = torch.nn.BCEWithLogitsLoss() # BCEWithLogitsLoss内部已经嵌入了sigmoid函数，并计算交叉熵损失

metric = matthews_corrcoef


# ------------------ STEP5 训练 -----------------
for epoch in range(100):
    for batch in train_loader:
        optimizer.zero_grad()
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        batch_idx = batch_idx.to(device)
        outputs = model(x, edge_index, batch_idx)
        outputs = outputs.squeeze(dim=-1) # 去掉最后一维
        loss = criterion(outputs, batch.y)
        loss.backward()
        optimizer.step()
        model.eval()

    with torch.no_grad():
        val_outputs, val_y = [], []
        for val_batch in test_loader:
            val_x, val_edge_index, val_batch_idx = val_batch.x, val_batch.edge_index, val_batch.batch
            val_batch_idx = val_batch_idx.to(device)
            y_pred = model(val_x, val_edge_index, val_batch_idx)
            y_pred = y_pred.squeeze(dim=-1)
            val_outputs.append(y_pred)
            val_y.append(val_batch.y)

    val_outputs = torch.cat(val_outputs).cpu()
    val_y = torch.cat(val_y).cpu()
    val_loss = criterion(val_outputs, val_y)
    
    preds = (val_outputs >= 0.5).to(int)
    val_metric = metric(val_y, preds)
    logger.info(
                f"Epoch: {epoch + 1:3d}/{epoch:3d} |"
                f" val loss: {val_loss:8.3f} | val metric: {val_metric:8.3f}"
            )
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')