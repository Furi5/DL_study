import torch_geometric.nn as geom_nn
from torch import nn

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv, # 经典 GCN 的卷积操作
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv, # 是广义的图卷积层，能够在图上执行加权的节点聚合操作
}


class GNNModel(nn.Module):
    def __init__(
        self, 
        D_in: int, 
        D_out: int,  
        D_hidden: int, 
        num_layers:int = 3,
        layer_name: str = "GCN",
        dp_rate: float = 0.1, 
        **kwargs  
    ):
        """
        Args:
            D_in (int): Number of input features
            D_hidden (int): Number of hidden features
            D_out (int): Number of output features
            num_layers (int): Number of hidden layers
            layer_name (str): Name of the GNN layer
            dp_rate (float): Dropout rate
            **kwargs: Additional keyword arguments
        表示函数或方法可以接受任意数量的关键字参数（keyword arguments）。
        ** 是解包操作符，它会将多个函数中没有默认的参数打包成一个字典传递给函数。
        """
        super().__init__() # 调用父类初始化函数
        gnn_layer = gnn_layer_by_name[layer_name]
        
        layers = []
        in_channels, out_channels = D_in, D_hidden
        
        # 构建模型
        for _ in range(num_layers):
            layers.append(gnn_layer(in_channels, out_channels, **kwargs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dp_rate))
            in_channels = D_hidden # 更新输入特征数
            
        layers.append(gnn_layer(in_channels, D_out, **kwargs))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): 每一个节点的特征
            edge_index (LongTensor): Graph edge indices
        """
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing): # 判断当前处理的 layer 是否是 MessagePassing 实例
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
        
        
