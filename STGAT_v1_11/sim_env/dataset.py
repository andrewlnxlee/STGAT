# dataset.py
import torch
import os
import numpy as np
from torch_geometric.data import Data, Dataset
from scipy.spatial.distance import cdist
import config

class RadarFileDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        self.root_dir = os.path.join(config.DATA_ROOT, split)
        self.file_list = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.npy')])
        self.conn_radius = 30.0
        
    def len(self):
        # 实际上我们这里返回的是 (Samples * Frames) 的总帧数，
        # 或者是 Samples 数。为了简单训练，我们把每个 episode 视为一个 batch 里的列表
        # 这里为了配合 PyG Loader，我们把每一帧都当作单独的图来训练
        # 注意：这里需要预先扫描总帧数，为了演示方便，我们简化为：
        # 一个 Dataset item = 一个 Episode (包含 50 帧)
        return len(self.file_list)

    def get(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        episode_data = np.load(file_path, allow_pickle=True)
        
        graph_list = []
        
        # 将一个 Episode 里的每一帧都转成图
        for frame in episode_data:
            if frame is None or len(frame['meas']) == 0:
                continue
                
            meas = frame['meas']
            labels = frame['labels']
            gt_centers = frame['gt_centers'] # [N_groups, 3] (id, x, y)
            
            # 1. 节点特征
            x = torch.tensor(meas, dtype=torch.float)
            
            # 2. 建图 (KNN / Radius)
            dist_mat = cdist(meas, meas)
            # src, dst = np.where((dist_mat < self.conn_radius) & (dist_mat > 0))
            # edge_index = torch.tensor([src, dst], dtype=torch.long)
            # 修改后 (加上 np.array 包装)
            src, dst = np.where((dist_mat < self.conn_radius) & (dist_mat > 0))
            # 关键修改：先转成 numpy array 再转 tensor
            edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
            # 3. 边属性
            pos_src = x[src]
            pos_dst = x[dst]
            rel_pos = pos_dst - pos_src
            dist = torch.norm(rel_pos, dim=1, keepdim=True)
            edge_attr = torch.cat([rel_pos, dist], dim=1)
            
            # 4. 边标签 (GT)
            l_src = labels[src]
            l_dst = labels[dst]
            edge_label = ((l_src == l_dst) & (l_src != 0)).astype(np.float32)
            edge_label = torch.tensor(edge_label)
            
            # 5. 封装
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                        edge_label=edge_label,
                        point_labels=torch.tensor(labels, dtype=torch.long),
                        gt_centers=torch.tensor(gt_centers, dtype=torch.float))
            graph_list.append(data)
            
        return graph_list # 返回一个列表，代表这一段时序数据