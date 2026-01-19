import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, BatchNorm

class GNNGroupTracker(nn.Module):
    def __init__(self, input_node_dim=2, input_edge_dim=3, hidden_dim=96): 
        # hidden_dim 设为 96 (能被多头整除，且拼接后不会太大)
        super(GNNGroupTracker, self).__init__()
        
        # --- 1. Embedding Layers ---
        self.node_encoder = nn.Sequential(
            nn.Linear(input_node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(), # 高级激活函数
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- 2. Multi-Scale Backbone (4 Layers) ---
        # 我们使用 4 层 Transformer，每层都带残差和归一化
        
        # Layer 1
        self.conv1 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.bn1 = BatchNorm(hidden_dim)
        
        # Layer 2
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.bn2 = BatchNorm(hidden_dim)
        
        # Layer 3
        self.conv3 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.bn3 = BatchNorm(hidden_dim)

        # Layer 4 (新增一层，提取更深语义)
        self.conv4 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.bn4 = BatchNorm(hidden_dim)
        
        # --- 3. Feature Fusion (Jumping Knowledge) ---
        # 最终特征维度 = 4层 * hidden_dim
        fusion_dim = hidden_dim * 4
        
        # 融合后的压缩层 (把拼接后的特征压回去，方便解码)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # --- 4. Decoders ---
        
        # 边分类 (利用融合特征)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        
        # 偏移回归 (利用融合特征)
        self.offset_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # --- Encoding ---
        h0 = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        
        # --- Layer 1 ---
        h1 = self.conv1(h0, edge_index, edge_attr=e)
        h1 = self.bn1(h1)
        h1 = F.gelu(h1)
        h1 = h1 + h0 # Residual
        
        # --- Layer 2 ---
        h2 = self.conv2(h1, edge_index, edge_attr=e)
        h2 = self.bn2(h2)
        h2 = F.gelu(h2)
        h2 = h2 + h1 # Residual
        
        # --- Layer 3 ---
        h3 = self.conv3(h2, edge_index, edge_attr=e)
        h3 = self.bn3(h3)
        h3 = F.gelu(h3)
        h3 = h3 + h2 # Residual
        
        # --- Layer 4 ---
        h4 = self.conv4(h3, edge_index, edge_attr=e)
        h4 = self.bn4(h4)
        h4 = F.gelu(h4)
        h4 = h4 + h3 # Residual
        
        # --- Jumping Knowledge Fusion (关键一步) ---
        # 将所有层的特征拼起来: [N, hidden*4]
        h_cat = torch.cat([h1, h2, h3, h4], dim=1)
        
        # 融合并压缩
        h_final = self.fusion_mlp(h_cat)
        
        # --- Decoding ---
        row, col = edge_index
        # 边的特征 = 源节点 + 目标节点 + 原始边属性
        edge_feat_cat = torch.cat([h_final[row], h_final[col], e], dim=1)
        edge_scores = self.edge_classifier(edge_feat_cat).squeeze(-1)
        
        pred_offsets = self.offset_regressor(h_final)
        
        return edge_scores, pred_offsets