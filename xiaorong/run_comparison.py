import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# 添加根目录路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import GNNGroupTracker  # 确保这里导入的是主模型
from xiaorong.ablation_model import AblationGNNTracker
from dataset import RadarFileDataset
from metrics import TrackingMetrics
from trackers.gnn_processor import GNNPostProcessor

def evaluate_variant(name, model, weight_path, is_full_model=False):
    device = torch.device(config.DEVICE)
    test_set = RadarFileDataset('test')
    
    if not os.path.exists(weight_path):
        print(f"Warning: {weight_path} not found for {name}.")
        return None
        
    try:
        # 核心加载逻辑：复刻 benchmark.py
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded {name} weights from {weight_path}")
    except Exception as e:
        print(f"Error loading {name} ({weight_path}): {str(e)[:200]}")
        return None
    
    model.to(device).eval()
    metrics = TrackingMetrics()
    
    for episode_idx in tqdm(range(len(test_set)), desc=f"Eval {name}"):
        episode_graphs = test_set.get(episode_idx)
        processor = GNNPostProcessor()
        
        for graph in episode_graphs:
            meas_points = graph.x.numpy()
            gt_data = graph.gt_centers.numpy()
            gt_centers = gt_data[:, 1:3] if len(gt_data) > 0 else np.zeros((0,2))
            gt_ids = gt_data[:, 0].astype(int) if len(gt_data) > 0 else []
            
            graph_dev = graph.to(device)
            with torch.no_grad():
                out = model(graph_dev)
                # 兼容旧版(2输出)和新版(3输出)
                scores, offsets = out[0], out[1]
            
            corrected_pos = meas_points + offsets.cpu().numpy()
            
            # 聚类逻辑：如果是 Full Model，尝试复刻 benchmark 中的联通分量逻辑，
            # 如果不通则回退到 DBSCAN
            labels = None
            if is_full_model:
                try:
                    mask = scores.cpu() > 0.5
                    edges = graph.edge_index.cpu()[:, mask].numpy()
                    if edges.shape[1] > 0:
                        adj = coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), 
                                       shape=(graph.num_nodes, graph.num_nodes))
                        _, labels = connected_components(adj, directed=False)
                except: labels = None

            if labels is None:
                try:
                    if len(corrected_pos) >= 3:
                        labels = DBSCAN(eps=30, min_samples=3).fit(corrected_pos).labels_
                    else:
                        labels = np.zeros(len(corrected_pos))
                except: labels = np.arange(len(corrected_pos))

            det_centers, det_shapes = [], []
            unique_labels = set(labels) if labels is not None else []
            for l in unique_labels:
                if l == -1: continue
                indices = np.where(labels == l)[0]
                if len(indices) < 2: continue # 过滤噪声
                det_centers.append(np.mean(corrected_pos[indices], axis=0))
                pts_raw = meas_points[indices]
                wh = np.percentile(pts_raw, 95, axis=0) - np.percentile(pts_raw, 5, axis=0)
                det_shapes.append(np.maximum(wh, 3.0))

            det_centers = np.array(det_centers).reshape(-1, 2)
            det_shapes = np.array(det_shapes).reshape(-1, 2)
            pred_c, pred_id, pred_sh = processor.update(det_centers, det_shapes if len(det_shapes)>0 else None)
            metrics.update(gt_centers, gt_ids, pred_c, pred_id)

    return metrics.compute()

if __name__ == "__main__":
    results = {}
    
    # 1. 评估 Full Model (复刻 benchmark.py 加载方式)
    full_model_path = config.MODEL_SAVE_PATH # 指向 best_model.pth
    print(f"--- Evaluating Full_Model (Baseline) ---")
    # 使用默认构造函数，不传 hidden_dim，让它使用主模型定义的默认值
    res_full = evaluate_variant("Full_Model", GNNGroupTracker(), full_model_path, is_full_model=True)
    if res_full: results["Full_Model"] = res_full

    # 2. 评估消融变体 (这些是你用 hidden_dim=64 训练的)
    h_dim = 64
    ablation_vars = {
        "No_Fourier": ({"use_fourier": False, "hidden_dim": h_dim}, "xiaorong/model_No_Fourier.pth"),
        "No_Adaptive_Fusion": ({"use_fourier": True, "fusion_mode": 'last', "hidden_dim": h_dim}, "xiaorong/model_No_Adaptive_Fusion.pth"),
        "Plain_GCN": ({"use_fourier": True, "use_transformer": False, "hidden_dim": h_dim}, "xiaorong/model_Plain_GCN.pth")
    }
    
    for name, (cfg, path) in ablation_vars.items():
        print(f"--- Evaluating {name} ---")
        res = evaluate_variant(name, AblationGNNTracker(**cfg), path)
        if res: results[name] = res
        
    if results:
        df = pd.DataFrame(results).T
        print("\n" + "="*100 + "\nCOMPLETE COMPARISON RESULTS\n" + "="*100)
        print(df.to_string())
        df.to_csv("xiaorong/ablation_comparison_final.csv")
    else:
        print("No results generated. Check error messages above.")
