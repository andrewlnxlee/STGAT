# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GNNGroupTracker(nn.Module):
    def __init__(self, input_node_dim=2, input_edge_dim=3, hidden_dim=64):
        super(GNNGroupTracker, self).__init__()
        
        # Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Processor (GAT)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, heads=4, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, heads=4, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, heads=4, concat=False)
        
        # Decoder 1: Edge Classification
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        
        # Decoder 2: Centroid Offset Regression
        self.offset_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        
        # Residual connections
        h = h + F.relu(self.conv1(h, edge_index, edge_attr=e))
        h = h + F.relu(self.conv2(h, edge_index, edge_attr=e))
        h = h + F.relu(self.conv3(h, edge_index, edge_attr=e))
        
        # Edge Prediction
        row, col = edge_index
        edge_feat = torch.cat([h[row], h[col], e], dim=1)
        edge_scores = self.edge_classifier(edge_feat).squeeze(-1)
        
        # Offset Prediction
        pred_offsets = self.offset_regressor(h)
        
        return edge_scores, pred_offsets