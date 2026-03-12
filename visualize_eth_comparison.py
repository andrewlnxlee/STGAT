import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN

import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from torch_geometric.data import Dataset 

# 导入所有跟踪器
from trackers.baseline import BaselineTracker
from trackers.gm_cphd import GMCPHDTracker
from trackers.cbmember import CBMeMBerTracker
from trackers.graph_mb import GraphMBTracker 
from trackers.social_stgcnn_tracker import SocialSTGCNNTracker
from trackers.gnn_processor import GNNPostProcessor

class ETHRealDataset(RadarFileDataset):
    def __init__(self):
        Dataset.__init__(self)
        self.root_dir = os.path.join(config.DATA_ROOT, "test_eth_real")
        if os.path.exists(self.root_dir):
            self.file_list = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.npy')])
        else:
            self.file_list = []
        self.conn_radius = 30.0

def visualize_comparison():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    test_set = ETHRealDataset()
    
    if len(test_set) == 0:
        print("❌ 数据集为空，请先运行 prepare_eth.py")
        return

    # 1. 加载我们的模型
    gnn_model = GNNGroupTracker().to(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        gnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        gnn_model.eval()
        print("✅ GNN 模型加载成功")
    else:
        print("⚠️ 未找到模型权重")
        gnn_model = None

    # 2. 初始化所有跟踪器
    trackers = {
        'Ground Truth': None,
        'Baseline': BaselineTracker(eps=35, min_samples=3),
        'GM-CPHD': GMCPHDTracker(),
        'CBMeMBer': CBMeMBerTracker(),
        'Social-STGCNN': SocialSTGCNNTracker(scene='eth'),
        'Graph-MB': GraphMBTracker(),
        'H-GAT-GT (Ours)': GNNPostProcessor()
    }
    
    # 3. 运行一段序列并记录结果
    episode_idx = 0 # 选择第一段序列
    episode_graphs = test_set.get(episode_idx)
    
    # 限制帧数，避免生成时间过长且 GIF 文件过大
    max_frames = 200
    episode_graphs = episode_graphs[:max_frames]
    
    # 存储轨迹用于绘图: {algo_name: [ {frame_idx: {id: [pos1, pos2, ...]}} ]}
    # 简化存储: {algo_name: [ (centers, ids) for each frame ]}
    all_results = {name: [] for name in trackers.keys()}
    all_meas = []
    
    print(f"正在运行跟踪器进行对比 (序列长度: {len(episode_graphs)} 帧)...")
    for frame_idx, graph in enumerate(tqdm(episode_graphs)):
        meas_points = graph.x.numpy()
        all_meas.append(meas_points)
        
        # GT
        gt_data = graph.gt_centers.numpy()
        gt_centers = gt_data[:, 1:3] if len(gt_data) > 0 else np.zeros((0,2))
        gt_ids = gt_data[:, 0].astype(int) if len(gt_data) > 0 else []
        all_results['Ground Truth'].append((gt_centers, gt_ids))

        # 运行各个算法
        # --- Baseline ---
        bc, bid, _ = trackers['Baseline'].step(meas_points)
        all_results['Baseline'].append((bc, bid))

        # 预处理用于 RFS 算法
        base_dets = []
        if len(meas_points) > 0:
            dbl = DBSCAN(eps=35, min_samples=3).fit_predict(meas_points)
            for l in set(dbl):
                if l == -1: continue
                base_dets.append(np.mean(meas_points[dbl == l], axis=0))

        # --- GM-CPHD ---
        cc, cid = trackers['GM-CPHD'].step(base_dets)
        all_results['GM-CPHD'].append((cc, cid))

        # --- CBMeMBer ---
        mc, mid = trackers['CBMeMBer'].step(base_dets)
        all_results['CBMeMBer'].append((mc, mid))

        # --- Social-STGCNN ---
        sc, sid, _ = trackers['Social-STGCNN'].step(meas_points)
        all_results['Social-STGCNN'].append((sc, sid))

        # --- Graph-MB ---
        gc, gid, _ = trackers['Graph-MB'].step(meas_points)
        all_results['Graph-MB'].append((gc, gid))

        # --- H-GAT-GT (Ours) ---
        pred_c, pred_id = np.array([]), np.array([])
        if gnn_model:
            graph_dev = graph.to(device)
            with torch.no_grad():
                out = gnn_model(graph_dev)
                # 修复解包: model 返回 4 个值 (edge_scores, offsets, uncertainty, h_final)
                _, offsets, _, _ = out
            
            corrected = meas_points + offsets.cpu().numpy()
            labels = np.array([])
            if len(corrected) > 0:
                try: labels = DBSCAN(eps=30, min_samples=3).fit(corrected).labels_
                except: pass
            
            det_c, det_s = [], []
            if len(labels) > 0:
                for l in set(labels):
                    if l == -1: continue
                    idx = np.where(labels == l)[0]
                    det_c.append(np.mean(corrected[idx], axis=0)) 
                    pts_raw = meas_points[idx]
                    wh = np.percentile(pts_raw, 95, axis=0) - np.percentile(pts_raw, 5, axis=0) if len(pts_raw)>1 else np.array([0,0])
                    det_s.append(np.maximum(wh, 3.0))
            
            det_c = np.array(det_c).reshape(-1, 2)
            det_s = np.array(det_s).reshape(-1, 2) if len(det_s) > 0 else None
            pred_c, pred_id, _ = trackers['H-GAT-GT (Ours)'].update(det_c, det_s)
        
        all_results['H-GAT-GT (Ours)'].append((pred_c, pred_id))

    # 4. 创建动画
    print("正在生成动画...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    algo_names = list(all_results.keys())
    
    # 轨迹历史记录 {algo: {id: [points]}}
    history = {name: {} for name in algo_names}
    colors = plt.cm.get_cmap('tab20', 50)

    def update(frame_idx):
        for i, name in enumerate(algo_names):
            ax = axes[i]
            ax.clear()
            
            centers, ids = all_results[name][frame_idx]
            meas = all_meas[frame_idx]
            
            # 画原始点 (背景)
            ax.scatter(meas[:, 0], meas[:, 1], c='gray', s=5, alpha=0.3)
            
            # 更新并画轨迹
            for c, tid in zip(centers, ids):
                if tid not in history[name]: history[name][tid] = []
                history[name][tid].append(c)
                
                # 只保留最近 20 帧的轨迹
                traj = np.array(history[name][tid][-20:])
                color = colors(tid % 50)
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5)
                ax.scatter(c[0], c[1], color=color, s=40, edgecolors='black')
                ax.text(c[0]+2, c[1]+2, f"ID:{tid}", fontsize=8, color=color)

            ax.set_title(name, fontsize=14, fontweight='bold' if 'Ours' in name else 'normal')
            ax.set_xlim(0, 1000) # 根据数据范围调整
            ax.set_ylim(0, 1000)
            ax.set_aspect('equal')
            if frame_idx == 0:
                ax.grid(True, linestyle='--', alpha=0.5)

        # 最后一个面板放一段总结文字
        ax_info = axes[7]
        ax_info.clear()
        ax_info.axis('off')
        ax_info.text(0.1, 0.5, f"Frame: {frame_idx}\nSequence: ETH Zurich\nComparison of 7 Algorithms", 
                     fontsize=12, verticalalignment='center')
        
        return axes

    anim = FuncAnimation(fig, update, frames=len(episode_graphs), interval=100)
    save_path = "eth_comparison_results.gif"
    anim.save(save_path, writer='pillow')
    print(f"✅ 对比动画已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    visualize_comparison()
