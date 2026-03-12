import torch
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
# 必须导入 PyG 的 Dataset 基类
from torch_geometric.data import Dataset 

import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from metrics import TrackingMetrics

# 引入跟踪器
from trackers.baseline import BaselineTracker
from trackers.gm_cphd import GMCPHDTracker
from trackers.cbmember import CBMeMBerTracker
from trackers.graph_mb import GraphMBTracker 
from trackers.gnn_processor import GNNPostProcessor

# --- 修正后的 Dataset 类 ---
class ETHRealDataset(RadarFileDataset):
    def __init__(self):
        # 【关键修复】显式初始化 PyG 的 Dataset 基类，构建内部索引
        Dataset.__init__(self)
        
        # 强行指向 test_eth_real 文件夹
        self.root_dir = os.path.join(config.DATA_ROOT, "test_eth_real")
        
        # 防止文件夹不存在报错
        if os.path.exists(self.root_dir):
            self.file_list = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.npy')])
        else:
            self.file_list = []
            print(f"⚠️ 警告: 文件夹 {self.root_dir} 不存在，请先运行 prepare_eth.py")
            
        self.conn_radius = 30.0
        print(f"已加载 ETH 真实数据集: {self.root_dir} (共 {len(self.file_list)} 帧)")

def run_eth_evaluation():
    # ... (后面的代码保持不变)
    device = torch.device(config.DEVICE)
    test_set = ETHRealDataset()
    
    if len(test_set) == 0:
        print("❌ 数据集为空，请先运行 prepare_eth.py")
        return

    # 加载模型
    gnn_model = GNNGroupTracker().to(device)
    # ... (其余逻辑与之前发给你的 evaluate_eth.py 一致) ...
    # 为了完整性，建议你直接保留之前文件的 run_eth_evaluation 函数体
    # 只要改了上面的 class ETHRealDataset 即可
    
    if os.path.exists(config.MODEL_SAVE_PATH):
        gnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        gnn_model.eval()
        print("✅ GNN 模型加载成功")
    else:
        print("⚠️ 未找到模型权重，GNN 将跳过或使用随机权重")
        gnn_model = None

    # 初始化跟踪器
    trackers = {
        'Baseline': BaselineTracker(eps=35, min_samples=3),
        'GM-CPHD': GMCPHDTracker(),
        'CBMeMBer': CBMeMBerTracker(),
        'Graph-MB': GraphMBTracker(),
        'H-GAT-GT': GNNPostProcessor() 
    }
    
    metrics = {name: TrackingMetrics() for name in trackers.keys()}
    
    print("开始基于真实轨迹的评估...")
    
    for episode_idx in tqdm(range(len(test_set))):
        episode_graphs = test_set.get(episode_idx)
        
        # 重置跟踪器
        for trk in trackers.values():
            if hasattr(trk, 'reset'): trk.reset()
            if isinstance(trk, GNNPostProcessor): trackers['H-GAT-GT'] = GNNPostProcessor()

        for graph in episode_graphs:
            # 准备数据
            gt_data = graph.gt_centers.numpy()
            gt_centers = gt_data[:, 1:3] if len(gt_data) > 0 else np.zeros((0,2))
            gt_ids = gt_data[:, 0].astype(int) if len(gt_data) > 0 else []
            meas_points = graph.x.numpy()
            
            # 获取 GT 形状 (用于 G-IoU)
            gt_shapes_arr = None
            gt_shapes_list = []
            pt_lbl = graph.point_labels.cpu().numpy()
            if len(gt_ids) > 0:
                for gid in gt_ids:
                    idx = np.where(pt_lbl == gid)[0]
                    if len(idx) > 1:
                        pts = meas_points[idx]
                        wh = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
                        gt_shapes_list.append(np.maximum(wh, 3.0))
                    else:
                        gt_shapes_list.append(np.array([3.0, 3.0]))
                gt_shapes_arr = np.array(gt_shapes_list).reshape(-1, 2)

            # --- 1. H-GAT-GT (Ours) ---
            t0 = time.time()
            pred_c, pred_id = np.array([]), np.array([])
            pred_shapes = None
            pt_map_ours = np.full(len(meas_points), -1)

            if gnn_model:
                graph_dev = graph.to(device)
                with torch.no_grad():
                    out = gnn_model(graph_dev)
                    if isinstance(out, tuple): _, offsets, _ = out
                    else: _, offsets = out
                
                corrected = meas_points + offsets.cpu().numpy()
                
                # 聚类
                labels = np.array([])
                if len(corrected) > 0:
                    try: labels = DBSCAN(eps=30, min_samples=3).fit(corrected).labels_
                    except: pass
                
                det_c, det_s, det_indices = [], [], []
                if len(labels) > 0:
                    for l in set(labels):
                        if l == -1: continue
                        idx = np.where(labels == l)[0]
                        det_indices.append(idx)
                        det_c.append(np.mean(corrected[idx], axis=0)) 
                        
                        pts_raw = meas_points[idx]
                        if len(pts_raw) > 1:
                            wh = np.percentile(pts_raw, 95, axis=0) - np.percentile(pts_raw, 5, axis=0)
                        else:
                            wh = np.array([0.,0.])
                        det_s.append(np.maximum(wh, 3.0))
                
                det_c = np.array(det_c).reshape(-1, 2)
                det_s = np.array(det_s).reshape(-1, 2) if len(det_s) > 0 else None
                
                if len(det_c) > 0:
                    pred_c, pred_id, pred_shapes = trackers['H-GAT-GT'].update(det_c, det_s)
                else:
                    pred_c, pred_id, pred_shapes = trackers['H-GAT-GT'].update(np.empty((0,2)), None)
                    
                if len(pred_c) > 0 and len(det_c) > 0:
                    cost = euclidean_distances(pred_c, det_c)
                    r, c = linear_sum_assignment(cost)
                    for ri, ci in zip(r, c):
                        if cost[ri, ci] < 20.0:
                            pt_map_ours[det_indices[ci]] = pred_id[ri]

            metrics['H-GAT-GT'].update_time(time.time() - t0)
            metrics['H-GAT-GT'].update(gt_centers, gt_ids, pred_c, pred_id, gt_shapes=gt_shapes_arr, pred_shapes=pred_shapes)
            metrics['H-GAT-GT'].update_clustering_metrics(graph.point_labels.cpu().numpy(), pt_map_ours)

            # --- 预处理 (为其他算法) ---
            t_pre_start = time.time()
            base_dets, base_map = [], {}
            if len(meas_points) > 0:
                dbl = DBSCAN(eps=35, min_samples=3).fit_predict(meas_points)
                valid_l = [l for l in set(dbl) if l != -1]
                for i, l in enumerate(valid_l):
                    idx = np.where(dbl == l)[0]
                    base_dets.append(np.mean(meas_points[idx], axis=0))
                    base_map[i] = idx
            pre_time = time.time() - t_pre_start
            
            def map_rfs(rfs_c, rfs_id):
                pm = np.full(len(meas_points), -1)
                if len(rfs_c) > 0 and len(base_dets) > 0:
                    cost = euclidean_distances(rfs_c, base_dets)
                    r, c = linear_sum_assignment(cost)
                    for ri, ci in zip(r, c):
                        if cost[ri, ci] < 20.0:
                            if ci in base_map: pm[base_map[ci]] = rfs_id[ri]
                return pm

            # --- 2. Baseline ---
            t0 = time.time()
            bc, bid, bmap = trackers['Baseline'].step(meas_points)
            metrics['Baseline'].update_time(time.time() - t0)
            metrics['Baseline'].update(gt_centers, gt_ids, bc, bid)
            metrics['Baseline'].update_clustering_metrics(graph.point_labels.cpu().numpy(), bmap)

            # --- 3. GM-CPHD ---
            t0 = time.time()
            cc, cid = trackers['GM-CPHD'].step(base_dets)
            metrics['GM-CPHD'].update_time(time.time() - t0 + pre_time)
            metrics['GM-CPHD'].update(gt_centers, gt_ids, cc, cid)
            metrics['GM-CPHD'].update_clustering_metrics(graph.point_labels.cpu().numpy(), map_rfs(cc, cid))

            # --- 4. CBMeMBer ---
            t0 = time.time()
            mc, mid = trackers['CBMeMBer'].step(base_dets)
            metrics['CBMeMBer'].update_time(time.time() - t0 + pre_time)
            metrics['CBMeMBer'].update(gt_centers, gt_ids, mc, mid)
            metrics['CBMeMBer'].update_clustering_metrics(graph.point_labels.cpu().numpy(), map_rfs(mc, mid))

            # --- 5. Graph-MB ---
            t0 = time.time()
            gc, gid, gmap = trackers['Graph-MB'].step(meas_points)
            metrics['Graph-MB'].update_time(time.time() - t0)
            metrics['Graph-MB'].update(gt_centers, gt_ids, gc, gid)
            metrics['Graph-MB'].update_clustering_metrics(graph.point_labels.cpu().numpy(), gmap)

    # --- 输出结果 ---
    res = {k: v.compute() for k, v in metrics.items()}
    df = pd.DataFrame(res).T
    cols = ['MOTA', 'G-IoU', 'OSPA (Total)', 'IDSW', 'RMSE (Pos)', 'Purity', 'Time']
    print("\n" + "="*80)
    print("ETH Zurich Dataset (Real Trajectory) Evaluation")
    print("="*80)
    print(df[cols].to_string())
    print("="*80)

if __name__ == "__main__":
    run_eth_evaluation()