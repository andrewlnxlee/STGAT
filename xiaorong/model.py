import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, BatchNorm
import numpy as np

# --- 模块 1: 傅里叶位置编码 (提升空间感知力) ---
class FourierFeatureEncoder(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        # 随机高斯矩阵 B，固定不更新，用于映射到高维
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x: [N, 2] -> [N, mapping_size]
        # proj: [N, mapping_size]
        x_proj = (2 * np.pi * x) @ self.B
        # output: [N, mapping_size * 2] (sin + cos)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# --- 模块 2: 自适应层融合 (替代简单的 Concat) ---
class AdaptiveLayerFusion(nn.Module):
    def __init__(self, hidden_dim, num_layers=4):
        super().__init__()
        # 注意力向量，用于计算每一层的重要性
        self.attn_vector = nn.Parameter(torch.randn(num_layers, hidden_dim))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, layers_list):
        # layers_list: [h1, h2, h3, h4], each is [N, hidden_dim]
        # stack: [4, N, hidden_dim]
        stack = torch.stack(layers_list, dim=0)
        
        # 计算注意力权重
        # weights: [4, 1, hidden_dim] -> 这里简化为对每个通道不同层加权
        # 或者是更简单的：学习 4 个标量。这里我们用稍微高级一点的：基于内容的注意力
        
        # 简单版：直接学习一组静态权重 alpha
        alpha = self.softmax(torch.mean(self.attn_vector, dim=1)) # [4]
        
        # 加权求和
        # out: [N, hidden_dim]
        out = 0
        for i in range(len(layers_list)):
            out += layers_list[i] * alpha[i]
            
        return out

class GNNGroupTracker(nn.Module):
    def __init__(self, input_node_dim=2, input_edge_dim=3, hidden_dim=96): 
        super(GNNGroupTracker, self).__init__()
        
        # 1. 增强型输入编码
        # 节点坐标映射: 2 -> 64
        self.fourier_dim = 64
        self.pos_encoder = FourierFeatureEncoder(input_node_dim, self.fourier_dim // 2, scale=2.0)
        
        # 节点特征融合: (原始2 + 傅里叶64) -> hidden
        self.node_mlp = nn.Sequential(
            nn.Linear(input_node_dim + self.fourier_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Multi-Scale Backbone (4 Layers)
        # 使用 TransformerConv
        self.conv1 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.conv3 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.bn3 = BatchNorm(hidden_dim)

        self.conv4 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.bn4 = BatchNorm(hidden_dim)
        
        # 3. 自适应融合 (Adaptive Fusion)
        # 替代之前的 concat，维度保持为 hidden_dim，参数量更小，表达更精准
        self.fusion_layer = AdaptiveLayerFusion(hidden_dim, num_layers=4)
        
        # 4. Decoders
        
        # 边分类
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # src + dst + edge_attr
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        
        # 偏移回归
        self.offset_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4) # dx, dy, sigma_x, sigma_y
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # --- 1. Enhanced Encoding ---
        # 傅里叶特征
        x_fourier = self.pos_encoder(x) # [N, 64]
        # 拼接原始坐标
        x_in = torch.cat([x, x_fourier], dim=1) # [N, 66]
        h0 = self.node_mlp(x_in)
        
        e = self.edge_encoder(edge_attr)
        
        # --- 2. Backbone ---
        # Layer 1
        h1 = self.conv1(h0, edge_index, edge_attr=e)
        h1 = self.bn1(h1)
        h1 = F.gelu(h1)
        h1 = h1 + h0 # Residual
        
        # Layer 2
        h2 = self.conv2(h1, edge_index, edge_attr=e)
        h2 = self.bn2(h2)
        h2 = F.gelu(h2)
        h2 = h2 + h1 
        
        # Layer 3
        h3 = self.conv3(h2, edge_index, edge_attr=e)
        h3 = self.bn3(h3)
        h3 = F.gelu(h3)
        h3 = h3 + h2 
        
        # Layer 4
        h4 = self.conv4(h3, edge_index, edge_attr=e)
        h4 = self.bn4(h4)
        h4 = F.gelu(h4)
        h4 = h4 + h3 
        
        # --- 3. Fusion ---
        # 使用自适应融合，而不是 Concat
        h_final = self.fusion_layer([h1, h2, h3, h4])
        
        # --- 4. Decoding ---
        row, col = edge_index
        # Edge Feat = Node_src + Node_dst + Edge_attr
        edge_feat_cat = torch.cat([h_final[row], h_final[col], e], dim=1)
        
        edge_scores = self.edge_classifier(edge_feat_cat).squeeze(-1)
        
        out = self.offset_regressor(h_final)
        pred_offsets = out[:, :2]   
        pred_uncertainty = F.softplus(out[:, 2:]) + 1e-6 # 保证方差为正
        
        return edge_scores, pred_offsets, pred_uncertainty