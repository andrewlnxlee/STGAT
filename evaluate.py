import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import time  # 引入时间模块
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import euclidean_distances

import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from metrics import TrackingMetrics

# Trackers
from trackers.baseline import BaselineTracker
from trackers.gnn_processor import GNNPostProcessor
from trackers.gm_cphd import GMCPHDTracker
from trackers.cbmember import CBMeMBerTracker

def run_evaluation():
    device = torch.device(config.DEVICE)
    print(f"Loading Test Data from {config.DATA_ROOT}...")
    test_set = RadarFileDataset('test')
    if len(test_set) == 0: return

    # Load Models
    gnn_model = GNNGroupTracker().to(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        gnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        gnn_model.eval()
    else: gnn_model = None

    # Trackers
    baseline_tracker = BaselineTracker(eps=35, min_samples=3)
    cphd_tracker = GMCPHDTracker()
    cbmember_tracker = CBMeMBerTracker()
    
    # Metrics
    metrics = {
        'Baseline': TrackingMetrics(),
        'GM-CPHD': TrackingMetrics(),
        'CBMeMBer': TrackingMetrics(),
        'H-GAT-GT': TrackingMetrics()
    }
    
    print("Running Evaluation Loop...")
    for episode_idx in tqdm(range(len(test_set))):
        episode_graphs = test_set.get(episode_idx)
        
        # Reset Trackers
        gnn_processor = GNNPostProcessor()
        baseline_tracker.reset()
        cphd_tracker.reset()
        cbmember_tracker.reset()
        
        for graph in episode_graphs:
            gt_data = graph.gt_centers.numpy()
            gt_centers = gt_data[:, 1:3] if len(gt_data) > 0 else np.zeros((0,2))
            gt_ids = gt_data[:, 0].astype(int) if len(gt_data) > 0 else []
            meas_points = graph.x.numpy()
            
            # --- 1. H-GAT-GT (Ours) ---
            t0 = time.time() # Start Timer
            if gnn_model:
                point_to_track_map_gnn = np.full(len(meas_points), -1)
                pred_c_gnn, pred_id_gnn = [], []
                
                if graph.edge_index.shape[1] > 0:
                    graph_dev = graph.to(device)
                    with torch.no_grad():
                        scores, offsets = gnn_model(graph_dev)
                    
                    mask = scores.cpu() > 0.5
                    edges = graph.edge_index.cpu()[:, mask].numpy()
                    
                    if edges.shape[1] > 0:
                        adj = coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(graph.num_nodes, graph.num_nodes))
                        _, labels = connected_components(adj, directed=False)
                    else:
                        labels = np.arange(graph.num_nodes)
                    
                    corrected_pos = meas_points + offsets.cpu().numpy()
                    det_centers = []
                    cluster_map = {}
                    
                    for l in set(labels):
                        if np.sum(labels == l) >= 3:
                            cluster_map[len(det_centers)] = l
                            det_centers.append(np.mean(corrected_pos[labels == l], axis=0))
                    
                    assignment = gnn_processor.update(det_centers)
                    pred_c_gnn = [det_centers[d_idx] for d_idx in assignment.keys()]
                    pred_id_gnn = list(assignment.values())
                    
                    for d_idx, t_id in assignment.items():
                        l = cluster_map[d_idx]
                        pt_idx = np.where(labels == l)[0]
                        point_to_track_map_gnn[pt_idx] = t_id
            
            t1 = time.time()
            metrics['H-GAT-GT'].update_time(t1 - t0)
            metrics['H-GAT-GT'].update(gt_centers, gt_ids, pred_c_gnn, pred_id_gnn)
            metrics['H-GAT-GT'].update_clustering_metrics(graph.point_labels.cpu().numpy(), point_to_track_map_gnn)

            # Pre-process for Baselines
            t_pre = time.time()
            if len(meas_points) > 0:
                db_labels = DBSCAN(eps=35, min_samples=3).fit_predict(meas_points)
                detected_centroids = []
                # Simple centroid map for purity
                centroid_to_points = {}
                valid_l = [l for l in set(db_labels) if l != -1]
                for i, l in enumerate(valid_l):
                    idx = np.where(db_labels == l)[0]
                    detected_centroids.append(np.mean(meas_points[idx], axis=0))
                    centroid_to_points[i] = idx
            else:
                detected_centroids = []
                centroid_to_points = {}
            t_pre_end = time.time()
            pre_time = t_pre_end - t_pre

            # --- 2. Baseline ---
            t0 = time.time()
            base_c, base_id, base_pt_lbl = baseline_tracker.step(meas_points)
            t1 = time.time()
            metrics['Baseline'].update_time(t1 - t0) # Baseline includes DBSCAN inside step
            metrics['Baseline'].update(gt_centers, gt_ids, base_c, base_id)
            metrics['Baseline'].update_clustering_metrics(graph.point_labels.cpu().numpy(), base_pt_lbl)
            
            # Helper for RFS Purity Mapping
            def get_rfs_point_labels(rfs_c, rfs_id):
                pt_lbl = np.full(len(meas_points), -1)
                if len(rfs_c) > 0 and len(detected_centroids) > 0:
                    cost = euclidean_distances(rfs_c, detected_centroids)
                    r, c = linear_sum_assignment(cost)
                    for r_i, c_i in zip(r, c):
                        if cost[r_i, c_i] < 10.0:
                            tid = rfs_id[r_i]
                            if c_i in centroid_to_points:
                                pt_lbl[centroid_to_points[c_i]] = tid
                return pt_lbl

            # --- 3. GM-CPHD ---
            t0 = time.time()
            cphd_c, cphd_id = cphd_tracker.step(detected_centroids)
            t1 = time.time()
            metrics['GM-CPHD'].update_time(t1 - t0 + pre_time) # Add clustering time
            metrics['GM-CPHD'].update(gt_centers, gt_ids, cphd_c, cphd_id)
            metrics['GM-CPHD'].update_clustering_metrics(graph.point_labels.cpu().numpy(), get_rfs_point_labels(cphd_c, cphd_id))
            
            # --- 4. CBMeMBer ---
            t0 = time.time()
            cb_c, cb_id = cbmember_tracker.step(detected_centroids)
            t1 = time.time()
            metrics['CBMeMBer'].update_time(t1 - t0 + pre_time)
            metrics['CBMeMBer'].update(gt_centers, gt_ids, cb_c, cb_id)
            metrics['CBMeMBer'].update_clustering_metrics(graph.point_labels.cpu().numpy(), get_rfs_point_labels(cb_c, cb_id))

    # Output Results
    final_results = {name: m.compute() for name, m in metrics.items()}
    df = pd.DataFrame(final_results).T
    
    # Reorder columns for better readability
    cols = ['MOTA', 'ID Switches', 'False Alarm Rate', 'OSPA (Total)', 'OSPA (Loc)', 'OSPA (Card)', 
            'RMSE (Pos)', 'Count Error', 'Purity', 'Completeness', 'Time (ms)']
    df = df[cols]
    
    print("\n" + "="*80)
    print("FINAL EXTENDED EVALUATION RESULTS")
    print("="*80)
    print(df.to_string())
    print("="*80)
    
    # Plotting
    plot_cats = ['MOTA', 'OSPA (Total)', 'OSPA (Loc)', 'OSPA (Card)', 'False Alarm Rate', 'Time (ms)']
    x = np.arange(len(plot_cats))
    width = 0.2
    fig, ax = plt.subplots(figsize=(16, 8))
    
    def get_vals(name):
        res = final_results[name]
        # Normalize for visualization if needed, but raw values are often better for tables
        # Here we just plot raw values. Note: ID Switch and Time can be large.
        return [res[c] for c in plot_cats]
    
    rects1 = ax.bar(x - 1.5*width, get_vals('Baseline'), width, label='Baseline')
    rects2 = ax.bar(x - 0.5*width, get_vals('GM-CPHD'), width, label='GM-CPHD')
    rects3 = ax.bar(x + 0.5*width, get_vals('CBMeMBer'), width, label='CBMeMBer')
    rects4 = ax.bar(x + 1.5*width, get_vals('H-GAT-GT'), width, label='H-GAT-GT')
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Comprehensive Benchmark (Lower is better for OSPA/FAR/Time)')
    ax.set_xticks(x); ax.set_xticklabels(plot_cats); ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1); autolabel(rects2); autolabel(rects3); autolabel(rects4)
    plt.tight_layout(); plt.savefig("final_benchmark_extended.png")
    print("Plot saved to final_benchmark_extended.png")

if __name__ == "__main__":
    run_evaluation()