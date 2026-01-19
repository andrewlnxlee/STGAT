# evaluate.py (已修复)
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import time
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import euclidean_distances

import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from metrics import TrackingMetrics

# --- Import Trackers ---
from trackers.baseline import BaselineTracker
from trackers.gm_cphd import GMCPHDTracker
from trackers.cbmember import CBMeMBerTracker
from trackers.graph_mb import GraphMBTracker 
from trackers.gnn_processor import GNNPostProcessor

def run_evaluation():
    device = torch.device(config.DEVICE)
    print(f"Loading Test Data from {config.DATA_ROOT}...")
    test_set = RadarFileDataset('test')
    if len(test_set) == 0: return

    # Load GNN
    gnn_model = GNNGroupTracker().to(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        gnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        gnn_model.eval()
    else: gnn_model = None

    # Instantiate Trackers
    # ... (这部分不变) ...
    baseline_tracker = BaselineTracker(eps=35, min_samples=3)
    gm_cphd_tracker = GMCPHDTracker()
    cbmember_tracker = CBMeMBerTracker()
    graph_mb_tracker = GraphMBTracker()
    metrics = {
        'Baseline (DBSCAN+KF)': TrackingMetrics(), 'GM-CPHD (Standard)': TrackingMetrics(),
        'CBMeMBer (Standard)': TrackingMetrics(), 'Graph-MB (Paper)': TrackingMetrics(),
        'H-GAT-GT (Ours)': TrackingMetrics()
    }
    
    print("Running Evaluation Loop (5 Trackers)...")
    for episode_idx in tqdm(range(len(test_set))):
        episode_graphs = test_set.get(episode_idx)
        
        # Reset trackers
        gnn_processor = GNNPostProcessor()
        baseline_tracker.reset(); gm_cphd_tracker.reset(); cbmember_tracker.reset(); graph_mb_tracker.reset()
        
        for graph in episode_graphs:
            gt_data = graph.gt_centers.numpy()
            gt_centers = gt_data[:, 1:3] if len(gt_data) > 0 else np.zeros((0,2))
            gt_ids = gt_data[:, 0].astype(int) if len(gt_data) > 0 else []
            meas_points = graph.x.numpy()
            
            # --- 1. H-GAT-GT (Ours) ---
            t0 = time.time()
            pred_c_gnn, pred_id_gnn = np.array([]), np.array([])
            point_to_track_map_gnn = np.full(len(meas_points), -1)
            pred_shapes_gnn = None 

            if gnn_model:
                graph_dev = graph.to(device)
                with torch.no_grad():
                    # 确保模型输出对应
                    model_output = gnn_model(graph_dev)
                    scores, offsets, uncertainty = model_output[0], model_output[1], model_output[2]
                
                corrected_pos = meas_points + offsets.cpu().numpy()
                
                labels = np.array([])
                if len(corrected_pos) > 0:
                    try: labels = DBSCAN(eps=30, min_samples=3).fit(corrected_pos).labels_
                    except: pass

                det_centers, det_shapes = [], []
                cluster_indices_list = [] 
                
                if len(labels) > 0:
                    for l in set(labels):
                        if l == -1: continue
                        indices = np.where(labels == l)[0]
                        cluster_indices_list.append(indices)
                        
                        det_centers.append(np.mean(corrected_pos[indices], axis=0))
                        
                        # --- FIX 2: 使用百分位数计算形状，抵抗离群点 ---
                        pts = corrected_pos[indices]
                        if len(pts) > 1:
                            lower = np.percentile(pts, 5, axis=0)
                            upper = np.percentile(pts, 95, axis=0)
                            wh = upper - lower
                        else:
                            wh = np.array([0., 0.])
                        det_shapes.append(np.maximum(wh, 3.0)) # 保底 3m 宽高

                det_centers = np.array(det_centers).reshape(-1, 2)
                det_shapes = np.array(det_shapes).reshape(-1, 2)
                if len(det_shapes) == 0: det_shapes = None

                # Tracking
                if det_centers.shape[0] > 0:
                    pred_c_gnn, pred_id_gnn, pred_shapes_gnn = gnn_processor.update(det_centers, det_shapes)
                else:
                    pred_c_gnn, pred_id_gnn, pred_shapes_gnn = gnn_processor.update(np.empty((0, 2)), None)

                # --- FIX 1: 使用匈牙利匹配重建点云到Track的映射 ---
                if len(pred_c_gnn) > 0 and len(det_centers) > 0:
                    # 稳健地将 tracker 输出的中心点匹配回 DBSCAN 检测出的中心点
                    cost_matrix = euclidean_distances(pred_c_gnn, det_centers)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    for r, c in zip(row_ind, col_ind):
                        # 使用一个更合理的阈值，比如 20.0 米
                        if cost_matrix[r, c] < 20.0:
                            track_id = pred_id_gnn[r]
                            point_indices = cluster_indices_list[c]
                            point_to_track_map_gnn[point_indices] = track_id
            
            t1 = time.time()
            metrics['H-GAT-GT (Ours)'].update_time(t1 - t0)
            
            # GT Shapes (也使用百分位数，保持一致性)
            gt_shapes_list = []
            pt_lbl = graph.point_labels.cpu().numpy()
            for gid in gt_ids:
                idx = np.where(pt_lbl == gid)[0]
                if len(idx) > 1:
                    pts = meas_points[idx]
                    lower = np.percentile(pts, 5, axis=0)
                    upper = np.percentile(pts, 95, axis=0)
                    gt_shapes_list.append(np.maximum(upper - lower, 3.0))
                else:
                    gt_shapes_list.append(np.array([3.0, 3.0]))
            gt_shapes_arr = np.array(gt_shapes_list).reshape(-1, 2) if len(gt_shapes_list) > 0 else None

            metrics['H-GAT-GT (Ours)'].update(gt_centers, gt_ids, pred_c_gnn, pred_id_gnn, 
                                              gt_shapes=gt_shapes_arr, pred_shapes=pred_shapes_gnn)
            # 传递修复后的 point_to_track_map_gnn
            metrics['H-GAT-GT (Ours)'].update_clustering_metrics(graph.point_labels.cpu().numpy(), point_to_track_map_gnn)


            # --- 其他算法部分保持不变 ---
            # ... (代码省略) ...
            t_pre = time.time()
            if len(meas_points) > 0:
                db_labels = DBSCAN(eps=35, min_samples=3).fit_predict(meas_points)
                detected_centroids = []
                centroid_to_points = {}
                valid_l = [l for l in set(db_labels) if l != -1]
                for i, l in enumerate(valid_l):
                    idx = np.where(db_labels == l)[0]
                    detected_centroids.append(np.mean(meas_points[idx], axis=0))
                    centroid_to_points[i] = idx
            else: detected_centroids = []; centroid_to_points = {}
            t_pre_end = time.time()
            pre_time = t_pre_end - t_pre

            def get_rfs_point_labels(rfs_c, rfs_id):
                pt_lbl = np.full(len(meas_points), -1)
                if len(rfs_c) > 0 and len(detected_centroids) > 0:
                    cost = euclidean_distances(rfs_c, detected_centroids)
                    r, c = linear_sum_assignment(cost)
                    for r_i, c_i in zip(r, c):
                        if cost[r_i, c_i] < 20.0:
                            tid = rfs_id[r_i]
                            if c_i in centroid_to_points:
                                pt_lbl[centroid_to_points[c_i]] = tid
                return pt_lbl

            # --- 2. Baseline ---
            t0 = time.time()
            base_c, base_id, base_pt_lbl = baseline_tracker.step(meas_points)
            t1 = time.time()
            metrics['Baseline (DBSCAN+KF)'].update_time(t1 - t0)
            metrics['Baseline (DBSCAN+KF)'].update(gt_centers, gt_ids, base_c, base_id)
            metrics['Baseline (DBSCAN+KF)'].update_clustering_metrics(graph.point_labels.cpu().numpy(), base_pt_lbl)

            # --- 3. GM-CPHD ---
            t0 = time.time()
            scphd_c, scphd_id = gm_cphd_tracker.step(detected_centroids)
            t1 = time.time()
            metrics['GM-CPHD (Standard)'].update_time(t1 - t0 + pre_time)
            metrics['GM-CPHD (Standard)'].update(gt_centers, gt_ids, scphd_c, scphd_id)
            metrics['GM-CPHD (Standard)'].update_clustering_metrics(graph.point_labels.cpu().numpy(), get_rfs_point_labels(scphd_c, scphd_id))

            # --- 4. CBMeMBer ---
            t0 = time.time()
            cb_c, cb_id = cbmember_tracker.step(detected_centroids)
            t1 = time.time()
            metrics['CBMeMBer (Standard)'].update_time(t1 - t0 + pre_time)
            metrics['CBMeMBer (Standard)'].update(gt_centers, gt_ids, cb_c, cb_id)
            metrics['CBMeMBer (Standard)'].update_clustering_metrics(graph.point_labels.cpu().numpy(), get_rfs_point_labels(cb_c, cb_id))

            # --- 5. Graph-MB (Paper) ---
            t0 = time.time()
            gmb_c, gmb_id, gmb_pt_lbl = graph_mb_tracker.step(meas_points)
            t1 = time.time()
            metrics['Graph-MB (Paper)'].update_time(t1 - t0)
            metrics['Graph-MB (Paper)'].update(gt_centers, gt_ids, gmb_c, gmb_id)
            metrics['Graph-MB (Paper)'].update_clustering_metrics(graph.point_labels.cpu().numpy(), gmb_pt_lbl)
    
    # ... (结果输出和绘图部分不变) ...
    final_res = {k: v.compute() for k, v in metrics.items()}
    df = pd.DataFrame(final_res).T
    cols = ['MOTA', 'MOTP','IDSW', 'FAR', 'OSPA (Total)', 'OSPA (Loc)', 'OSPA (Card)', 
            'RMSE (Pos)', 'Count Err', 'Purity', 'Completeness','G-IoU', 'Time','Comp',]
    df = df[[c for c in cols if c in df.columns]]
    print("\n" + "="*100); print("FINAL 5-WAY COMPARISON RESULTS"); print("="*100)
    print(df.to_string()); print("="*100)

if __name__ == "__main__":
    run_evaluation()