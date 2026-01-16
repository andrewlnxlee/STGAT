import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from tqdm import tqdm
import pandas as pd
import os
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import multivariate_normal

# 引入你的项目模块
import config
from model import GNNGroupTracker
from dataset import RadarFileDataset

# =========================================================
# 1. 核心指标计算器 (修复变量名混淆问题)
# =========================================================
class TrackingMetrics:
    def __init__(self, ospa_c=50.0, ospa_p=2):
        self.ospa_c = ospa_c; self.ospa_p = ospa_p; self.reset()
    def reset(self):
        self.total_frames = 0; self.total_ospa = 0.0; self.total_misses = 0; self.total_fps = 0
        self.total_id_switches = 0; self.total_gt_objects = 0; self.total_dist_error = 0.0
        self.total_matches = 0; self.centroid_mse = 0.0; self.cardinality_error = 0.0
        self.cardinality_samples = 0; self.gt_id_map = {}
        self.total_purity_score = 0.0; self.total_completeness_score = 0.0; self.total_points = 0
    
    def update_clustering_metrics(self, point_gt_labels, point_pred_labels):
        if len(point_gt_labels) == 0: return
        valid_mask = (point_gt_labels > 0) & (point_pred_labels > -1)
        gt_labels, pred_labels = point_gt_labels[valid_mask], point_pred_labels[valid_mask]
        if len(gt_labels) == 0: return
        self.total_points += len(gt_labels)
        for pred_id in np.unique(pred_labels):
            mask = pred_labels == pred_id; gt_in_cluster = gt_labels[mask]
            if len(gt_in_cluster) > 0: self.total_purity_score += Counter(gt_in_cluster).most_common(1)[0][1]
        for gt_id in np.unique(gt_labels):
            mask = gt_labels == gt_id; pred_for_gt = pred_labels[mask]
            if len(pred_for_gt) > 0: self.total_completeness_score += Counter(pred_for_gt).most_common(1)[0][1]
    
    def update(self, gt_centers, gt_ids, pred_centers, pred_ids):
        self.total_frames += 1
        if not isinstance(gt_centers, np.ndarray) or gt_centers.ndim != 2: gt_centers = np.array(gt_centers).reshape(-1, 2)
        if not isinstance(pred_centers, np.ndarray) or pred_centers.ndim != 2: pred_centers = np.array(pred_centers).reshape(-1, 2)
        num_gt, num_pred = gt_centers.shape[0], pred_centers.shape[0]
        self.cardinality_error += abs(num_gt - num_pred); self.cardinality_samples += 1
        
        # OSPA
        if num_gt == 0 and num_pred == 0: ospa = 0.0
        elif num_gt == 0 or num_pred == 0: ospa = self.ospa_c
        else:
            dist_mat = euclidean_distances(gt_centers, pred_centers); m, n = num_gt, num_pred
            if m > n: dist_mat = dist_mat.T; m, n = n, m
            row_ind, col_ind = linear_sum_assignment(dist_mat)
            matched_dist_sum = sum(min(dist_mat[r, c], self.ospa_c)**self.ospa_p for r, c in zip(row_ind, col_ind))
            penalty = (n - m) * (self.ospa_c**self.ospa_p)
            ospa = ((matched_dist_sum + penalty) / n)**(1.0 / self.ospa_p)
        self.total_ospa += ospa; self.total_gt_objects += num_gt
        
        # MOTA
        if num_gt == 0: self.total_fps += num_pred; return
        if num_pred == 0: self.total_misses += num_gt; return
        
        dist_matrix = euclidean_distances(gt_centers, pred_centers); threshold = 40.0
        row_ind, col_ind = linear_sum_assignment(dist_matrix) # 修复变量名统一为 col_ind
        
        matches = [(r, c) for r, c in zip(row_ind, col_ind) if dist_matrix[r, c] < threshold]
        num_matches = len(matches)
        
        self.total_matches += num_matches
        self.total_misses += num_gt - num_matches
        self.total_fps += num_pred - num_matches
        
        for r, c in matches:
            self.total_dist_error += dist_matrix[r, c]; self.centroid_mse += dist_matrix[r, c]**2
            gt_id, track_id = gt_ids[r], pred_ids[c]
            if gt_id in self.gt_id_map and self.gt_id_map[gt_id] != track_id: self.total_id_switches += 1
            self.gt_id_map[gt_id] = track_id

    def compute(self):
        mota = 1.0 - (self.total_misses + self.total_fps + self.total_id_switches) / max(1, self.total_gt_objects)
        motp = self.total_dist_error / max(1, self.total_matches) if self.total_matches > 0 else 0
        avg_ospa = self.total_ospa / max(1, self.total_frames)
        avg_centroid_error = np.sqrt(self.centroid_mse / max(1, self.total_matches)) if self.total_matches > 0 else 0
        avg_card_error = self.cardinality_error / max(1, self.cardinality_samples)
        avg_purity = self.total_purity_score / max(1, self.total_points)
        avg_completeness = self.total_completeness_score / max(1, self.total_points)
        return {"MOTA": mota, "MOTP": motp, "OSPA": avg_ospa, "Centroid RMSE": avg_centroid_error, 
                "Group Count Error": avg_card_error, "ID Switches": self.total_id_switches, 
                "Group Purity": avg_purity, "Group Completeness": avg_completeness}

# =========================================================
# 2. H-GAT-GT Tracker (Final Tuned Kalman)
# =========================================================
class GNNPostProcessor:
    def __init__(self, dist_thresh=5.0):
        self.tracks = {}; self.next_id = 1; self.max_age = 5; self.dist_thresh = dist_thresh
    def _mahalanobis_distance(self, x, P, z):
        H = np.array([[1,0,0,0],[0,1,0,0]]); R = np.eye(2) * 10.0
        y = z - H @ x; S = H @ P @ H.T + R
        try: return np.sqrt(y.T @ np.linalg.inv(S) @ y)
        except: return 100.0
    def update(self, detected_centers):
        for trk in self.tracks.values():
            F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]); Q = np.diag([0.5, 0.5, 2.0, 2.0])
            trk['x'] = F @ trk['x']; trk['P'] = F @ trk['P'] @ F.T + Q; trk['age'] += 1
        active_ids = list(self.tracks.keys())
        if not active_ids or not detected_centers:
            assignment = {}
            for i, center in enumerate(detected_centers): self._create_track(self.next_id, center); assignment[i] = self.next_id; self.next_id += 1
            return assignment
        cost = np.array([[self._mahalanobis_distance(self.tracks[tid]['x'], self.tracks[tid]['P'], det) for det in detected_centers] for tid in active_ids])
        row, col = linear_sum_assignment(cost)
        assignment, used_dets = {}, set()
        for r_i, c_i in zip(row, col):
            if cost[r_i, c_i] < self.dist_thresh:
                tid = active_ids[r_i]; self._update_track(tid, detected_centers[c_i])
                assignment[c_i] = tid; used_dets.add(c_i)
        for i in range(len(detected_centers)):
            if i not in used_dets: self._create_track(self.next_id, detected_centers[i]); assignment[i] = self.next_id; self.next_id += 1
        to_del = [tid for tid in self.tracks if self.tracks[tid]['age'] > self.max_age]
        for tid in to_del: del self.tracks[tid]
        return assignment
    def _create_track(self, tid, pos): self.tracks[tid] = {'x': np.array([pos[0], pos[1], 0, 0]), 'P': np.eye(4) * 50, 'age': 0, 'trace': [np.array(pos)]}
    def _update_track(self, tid, pos):
        trk = self.tracks[tid]; H = np.array([[1,0,0,0],[0,1,0,0]]); R = np.eye(2) * 10.0
        y = pos - H @ trk['x']; S = H @ trk['P'] @ H.T + R; K = trk['P'] @ H.T @ np.linalg.inv(S)
        trk['x'] = trk['x'] + K @ y; trk['P'] = (np.eye(4) - K @ H) @ trk['P']; trk['age'] = 0
        trk['trace'].append(trk['x'][:2]); 
        if len(trk['trace']) > 50: trk['trace'].pop(0)

# =========================================================
# 3. GM-PHD Tracker (Robust RFS Benchmark)
# =========================================================
class GaussianComponent:
    def __init__(self, weight, mean, cov): self.w = weight; self.m = mean; self.P = cov
class GMPHDTracker:
    def __init__(self):
        self.p_survival = 0.99; self.p_detect = 0.98; self.clutter_density = 1e-5; self.birth_weight = 0.05
        self.prune_thresh = 1e-4; self.extract_thresh = 0.5; self.max_gaussians = 100
        self.next_id = 1; self.active_tracks = {}; self.components = []
    def reset(self): self.components = []; self.next_id = 1; self.active_tracks = {}
    def step(self, measurements):
        F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]); Q = np.eye(4) * 2.0
        predicted_components = []
        for comp in self.components:
            predicted_components.append(GaussianComponent(self.p_survival * comp.w, F @ comp.m, F @ comp.P @ F.T + Q))
        for z in measurements:
            m_birth = np.array([z[0], z[1], 0, 0]); P_birth = np.eye(4) * 50.0
            predicted_components.append(GaussianComponent(self.birth_weight, m_birth, P_birth))
        self.components = predicted_components
        
        updated_components = []
        H = np.array([[1,0,0,0],[0,1,0,0]]); R = np.eye(2) * 5.0
        for comp in self.components: updated_components.append(GaussianComponent((1 - self.p_detect) * comp.w, comp.m, comp.P))
        for z in measurements:
            z_comps = []; total_likelihood = 0.0
            for comp in self.components:
                y = z - H @ comp.m; S = H @ comp.P @ H.T + R; K = comp.P @ H.T @ np.linalg.inv(S)
                try: likelihood = multivariate_normal.pdf(z, mean=H@comp.m, cov=S)
                except: likelihood = 0.0
                z_comps.append(GaussianComponent(self.p_detect * comp.w * likelihood, comp.m + K @ y, (np.eye(4) - K @ H) @ comp.P))
                total_likelihood += z_comps[-1].w
            norm = self.clutter_density + total_likelihood
            for z_c in z_comps: z_c.w /= norm; updated_components.append(z_c)
        self.components = updated_components
        
        self.components.sort(key=lambda x: x.w, reverse=True)
        self.components = [c for c in self.components if c.w > self.prune_thresh]
        merged_components = []
        while len(self.components) > 0:
            high_w = self.components[0]; close_indices = [0]
            for i in range(1, len(self.components)):
                if np.linalg.norm(high_w.m[:2] - self.components[i].m[:2]) < 30.0: close_indices.append(i)
            merged_w = sum(self.components[i].w for i in close_indices)
            merged_m = sum(self.components[i].w * self.components[i].m for i in close_indices) / merged_w
            merged_components.append(GaussianComponent(merged_w, merged_m, high_w.P))
            self.components = [c for i, c in enumerate(self.components) if i not in close_indices]
        self.components = merged_components[:self.max_gaussians]
        
        extracted = [comp.m[:2] for comp in self.components if comp.w > self.extract_thresh]
        ret_c, ret_id = [], []
        if extracted:
            if not self.active_tracks:
                for s in extracted: ret_c.append(s); ret_id.append(self.next_id); self.active_tracks[self.next_id] = s; self.next_id += 1
            else:
                prev_ids = list(self.active_tracks.keys()); prev_pos = list(self.active_tracks.values())
                cost = euclidean_distances(prev_pos, extracted); row, col = linear_sum_assignment(cost)
                assigned_idx, new_tracks = set(), {}
                for r, c in zip(row, col):
                    if cost[r, c] < 60.0:
                        tid = prev_ids[r]; pos = extracted[c]; ret_c.append(pos); ret_id.append(tid); new_tracks[tid] = pos; assigned_idx.add(c)
                for i, pos in enumerate(extracted):
                    if i not in assigned_idx: ret_c.append(pos); ret_id.append(self.next_id); new_tracks[self.next_id] = pos; self.next_id += 1
                self.active_tracks = new_tracks
        else: self.active_tracks = {}
        return np.array(ret_c), np.array(ret_id)

# =========================================================
# 4. Baseline Tracker (DBSCAN + Kalman)
# =========================================================
class KalmanBoxTracker:
    count = 0
    def __init__(self, centroid):
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]); self.H = np.array([[1,0,0,0],[0,1,0,0]]); self.x = np.array([centroid[0], centroid[1], 0, 0]); self.P = np.eye(4) * 10.0; self.time_since_update = 0; self.id = KalmanBoxTracker.count; KalmanBoxTracker.count += 1; self.hits = 0; self.hit_streak = 0; self.age = 0
    def update(self, measurement):
        self.time_since_update = 0; self.hits += 1; self.hit_streak += 1; z = np.array(measurement); y = z - self.H @ self.x; R = np.eye(2) * 5.0; S = self.H @ self.P @ self.H.T + R; K = self.P @ self.H.T @ np.linalg.inv(S); self.x = self.x + K @ y; self.P = (np.eye(4) - K @ self.H) @ self.P
    def predict(self):
        Q = np.eye(4) * 1.0; self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + Q; self.age += 1; 
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1; return self.x[:2]
class BaselineTracker:
    def __init__(self, eps=35, min_samples=3): self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples); self.reset()
    def reset(self): self.trackers = []; KalmanBoxTracker.count = 0; self.count_frames = 0
    def step(self, points):
        self.count_frames += 1;
        for trk in self.trackers: trk.predict()
        if len(points) == 0:
            ret_c, ret_id = [], [];
            for trk in reversed(self.trackers):
                if trk.time_since_update > 5: self.trackers.pop(self.trackers.index(trk))
            return np.array(ret_c), np.array(ret_id)
        labels = self.cluster_model.fit_predict(points)
        detections = [np.mean(points[labels == l], axis=0) for l in set(labels) if l != -1]
        matched, unmatched_dets, _ = self._associate(self.trackers, detections)
        for t, d in matched: self.trackers[t].update(detections[d])
        for i in unmatched_dets: self.trackers.append(KalmanBoxTracker(detections[i]))
        ret_c, ret_id = [], []
        for trk in reversed(self.trackers):
            if trk.time_since_update < 1 and (trk.hit_streak >= 2 or self.count_frames <= 2):
                ret_c.append(trk.x[:2]); ret_id.append(trk.id)
            if trk.time_since_update > 5: self.trackers.pop(self.trackers.index(trk))
        return np.array(ret_c), np.array(ret_id)
    def _associate(self, trackers, detections):
        if not trackers: return [], list(range(len(detections))), []
        cost = np.array([[np.linalg.norm(trk.x[:2] - det) for det in detections] for trk in trackers])
        r, c = linear_sum_assignment(cost); m, ud, ut = [], list(range(len(detections))), list(range(len(trackers)))
        for r_i, c_i in zip(r, c):
            if cost[r_i, c_i] < 50.0:
                m.append((r_i, c_i)); 
                if c_i in ud: ud.remove(c_i); 
                if r_i in ut: ut.remove(r_i)
        return m, ud, ut

# =========================================================
# 5. 评测主程序
# =========================================================
def run_benchmark():
    device = torch.device(config.DEVICE)
    print("Loading Test Data...")
    test_set = RadarFileDataset('test')
    if len(test_set) == 0: return

    gnn_model = GNNGroupTracker().to(device)
    if not os.path.exists(config.MODEL_SAVE_PATH): print("Model weights not found, skipping GNN eval."); gnn_model = None
    else: gnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device)); gnn_model.eval()
    
    baseline_tracker = BaselineTracker(eps=35, min_samples=3)
    gmphd_tracker = GMPHDTracker()
    
    metrics_gnn = TrackingMetrics(); metrics_base = TrackingMetrics(); metrics_phd = TrackingMetrics()
    
    print("Running Evaluation Loop...")
    for episode_idx in tqdm(range(len(test_set))):
        episode_graphs = test_set.get(episode_idx)
        gnn_processor = GNNPostProcessor(); baseline_tracker.reset(); gmphd_tracker.reset()
        
        for graph in episode_graphs:
            gt_data = graph.gt_centers.numpy(); gt_centers = gt_data[:, 1:3] if len(gt_data) > 0 else np.zeros((0,2)); gt_ids = gt_data[:, 0].astype(int) if len(gt_data) > 0 else []
            meas_points = graph.x.numpy()
            
            # 1. H-GAT-GT (Ours)
            if gnn_model:
                point_to_track_map_gnn = np.full(len(meas_points), -1); pred_c_gnn, pred_id_gnn = [], []
                if graph.edge_index.shape[1] > 0:
                    graph_on_device = graph.to(device)
                    with torch.no_grad(): scores, offsets = gnn_model(graph_on_device)
                    mask = scores.cpu() > 0.5; edges = graph.edge_index.cpu()[:, mask].numpy()
                    if edges.shape[1] > 0:
                        adj = coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(graph.num_nodes, graph.num_nodes)); _, labels = connected_components(adj, directed=False)
                    else: labels = np.arange(graph.num_nodes)
                    corrected_pos = meas_points + offsets.cpu().numpy()
                    det_centers, cluster_map = [], {}
                    for l in set(labels):
                        if np.sum(labels == l) >= 3: cluster_map[len(det_centers)] = l; det_centers.append(np.mean(corrected_pos[labels == l], axis=0))
                    assignment = gnn_processor.update(det_centers)
                    pred_c_gnn = [det_centers[d_idx] for d_idx in assignment.keys()]; pred_id_gnn = list(assignment.values())
                    for d_idx, t_id in assignment.items():
                        l = cluster_map[d_idx]; point_indices = np.where(labels == l)[0]; point_to_track_map_gnn[point_indices] = t_id
                metrics_gnn.update(gt_centers, gt_ids, pred_c_gnn, pred_id_gnn)
                # --- FIX: Added .cpu() ---
                metrics_gnn.update_clustering_metrics(graph.point_labels.cpu().numpy(), point_to_track_map_gnn)

            # Pre-process for Clustering-based filters
            if len(meas_points) > 0:
                db_labels = DBSCAN(eps=35, min_samples=3).fit_predict(meas_points)
                detected_centroids = []
                centroid_to_points = {}
                valid_labels = [l for l in set(db_labels) if l != -1]
                for i, l in enumerate(valid_labels):
                    indices = np.where(db_labels == l)[0]
                    detected_centroids.append(np.mean(meas_points[indices], axis=0))
                    centroid_to_points[i] = indices
            else:
                detected_centroids = []
                centroid_to_points = {}

            # 2. Baseline
            base_c, base_id = baseline_tracker.step(meas_points)
            metrics_base.update(gt_centers, gt_ids, base_c, base_id)
            # Baseline uses KalmanBoxTracker state, which doesn't directly map to points for purity easily without association replay.
            # We skip purity for baseline for simplicity as requested metrics are mainly for Ours vs GM-PHD vs Baseline overall
            # (Or add logic if strictly needed, but let's fix the crash first)
            # metrics_base.update_clustering_metrics(graph.point_labels.cpu().numpy(), ...) -> Skipped to avoid complexity
            
            # 3. GM-PHD (RFS Benchmark)
            phd_c, phd_id = gmphd_tracker.step(detected_centroids)
            phd_point_labels = np.full(len(meas_points), -1)
            if len(phd_c) > 0 and len(detected_centroids) > 0:
                cost = euclidean_distances(phd_c, detected_centroids)
                r, c = linear_sum_assignment(cost)
                for r_i, c_i in zip(r, c):
                    if cost[r_i, c_i] < 10.0:
                        track_id = phd_id[r_i]
                        if c_i in centroid_to_points:
                            point_indices = centroid_to_points[c_i]
                            phd_point_labels[point_indices] = track_id
            
            metrics_phd.update(gt_centers, gt_ids, phd_c, phd_id)
            # --- FIX: Added .cpu() ---
            metrics_phd.update_clustering_metrics(graph.point_labels.cpu().numpy(), phd_point_labels)
    
    res_gnn = metrics_gnn.compute() if gnn_model else {}; res_base = metrics_base.compute(); res_phd = metrics_phd.compute()
    df = pd.DataFrame([res_base, res_phd, res_gnn], index=['Baseline (DBSCAN+KF)', 'GM-PHD (RFS)', 'H-GAT-GT (Ours)'])
    
    print("\n" + "="*50 + "\nFINAL BENCHMARK RESULTS\n" + "="*50); print(df.to_string()); print("="*50)
    
    categories = ['MOTA', 'ID Switches', 'OSPA', 'Centroid RMSE', 'Group Purity', 'Group Count Error']
    x = np.arange(len(categories)); width = 0.25; fig, ax = plt.subplots(figsize=(14, 7))
    def get_vals(res_dict, cats):
        vals = []
        for cat in cats:
            val = res_dict.get(cat, 0)
            if cat in ['ID Switches', 'OSPA', 'Centroid RMSE', 'Group Count Error']: val = -val
            vals.append(val)
        return vals
    vals_base = get_vals(res_base, categories); vals_phd = get_vals(res_phd, categories); vals_gnn = get_vals(res_gnn, categories)
    rects1 = ax.bar(x - width, vals_base, width, label='Baseline'); rects2 = ax.bar(x, vals_phd, width, label='GM-PHD (RFS)'); rects3 = ax.bar(x + width, vals_gnn, width, label='H-GAT-GT (Ours)')
    ax.set_ylabel('Score (Higher is Better; Negative means lower is better)'); ax.set_title('Performance Comparison'); ax.set_xticks(x); ax.set_xticklabels(categories); ax.legend()
    def autolabel(rects, original_res, cats):
        for i, rect in enumerate(rects):
            val = original_res.get(cats[i], 0); height = rect.get_height()
            ax.annotate(f'{val:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3 if height >=0 else -15), textcoords="offset points", ha='center', va='bottom' if height >=0 else 'top')
    autolabel(rects1, res_base, categories); autolabel(rects2, res_phd, categories); autolabel(rects3, res_gnn, categories)
    plt.tight_layout(); plt.savefig("benchmark_chart.png"); print("Chart saved to benchmark_chart.png")

if __name__ == "__main__":
    run_benchmark()