import torch
import numpy as np
import config
from model import GNNGroupTracker
from evaluate_ewap import EWAPDataset

device = torch.device('cpu')
gnn_model = GNNGroupTracker().to(device)
gnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
gnn_model.eval()

test_set = EWAPDataset('test_ewap_eth')
graph = test_set[0][0]

with torch.no_grad():
    _, offsets, _ = gnn_model(graph)

pts = graph.x.numpy()
labels = graph.point_labels.numpy()
gt_centers = graph.gt_centers.numpy()

print(f'\n--- 测试第一帧 ---')
print(f'原始点: \n{pts[:5]}')

for i in range(min(5, len(pts))):
    gid = labels[i]
    if gid > 0:
        c = gt_centers[gt_centers[:,0] == gid][0][1:3]
        real_offset = c - pts[i]
        print(f'标签: {gid}, 理想要走的偏移: {real_offset}, 模型给出的偏移 (Offsets): {offsets[i].numpy()}')
