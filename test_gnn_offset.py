import torch
import numpy as np
import config
from model import GNNGroupTracker
from evaluate_ewap import EWAPDataset

device = torch.device('cpu')
gnn_model = GNNGroupTracker().to(device)
gnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
gnn_model.eval()

# 读取 ETH
test_set = EWAPDataset('test_ewap_eth')
graph = test_set[0][0]

with torch.no_grad():
    _, offsets, _ = gnn_model(graph)

pts = graph.x.numpy()
labels = graph.point_labels.numpy()
gt_centers = graph.gt_centers.numpy()

print(f'\n================================')
print(f'H-GAT-GT 模型偏移值输出测试 (ETH)')
print(f'================================\n')

print(f"总点数: {len(pts)}")

for i in range(len(pts)):
    gid = labels[i]
    if gid > 0:
        c = gt_centers[gt_centers[:,0] == gid][0][1:3]
        real_offset = c - pts[i]
        gnn_offset = offsets[i].numpy()
        
        # 看看 GNN 是不是发疯了把人推散了
        print(f'行人 {i:02d} | 真实群体标签: {gid}')
        print(f'  当前位置: {pts[i]}')
        print(f'  距离真实群组中心的距离向量 (应该移动): {real_offset}')
        print(f'  GNN 输出的修正向量 (实际移动):      {gnn_offset}')
        print(f'  修正后位置: {pts[i] + gnn_offset}\n')
