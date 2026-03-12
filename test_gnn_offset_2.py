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

# 找一个行人比较多 (>3) 的帧
for ep_idx in range(len(test_set)):
    graphs = test_set[ep_idx]
    for frame_idx, graph in enumerate(graphs):
        pts = graph.x.numpy()
        if len(pts) > 3:
            with torch.no_grad():
                _, offsets, _ = gnn_model(graph)
            
            labels = graph.point_labels.numpy()
            gt_centers = graph.gt_centers.numpy()
            
            print(f'\n================================')
            print(f'H-GAT-GT 模型偏移 (Episode {ep_idx}, Frame {frame_idx})')
            print(f'================================\n')
            print(f"总点数: {len(pts)}")
            
            for i in range(len(pts)):
                gid = labels[i]
                if gid > 0:
                    c = gt_centers[gt_centers[:,0] == gid][0][1:3]
                    real_offset = c - pts[i]
                    gnn_offset = offsets[i].numpy()
                    
                    print(f'行人 {i:02d} | 群标签: {gid}')
                    print(f'  距离真实群组中心的距离向量 (理应移动): {real_offset}')
                    print(f'  GNN 输出的修正向量 (实际移动):      {gnn_offset}')
            exit(0)
