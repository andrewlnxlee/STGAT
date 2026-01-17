# trackers/graph_mb.py (Final "Capstone" Version)

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from scipy.linalg import cholesky

# --- UKF Parameters ---
N_DIM = 5; ALPHA = 0.1; BETA = 2.0; KAPPA = 0.0
LAMBDA = ALPHA**2 * (N_DIM + KAPPA) - N_DIM

# --- Tracker Parameters ---
PARENT_SEARCH_RADIUS = 50.0 
Q_PROCESS_BASE = np.diag([2.0, 2.0, 5.0, 5.0, 1e-3]) 
Q_COLLAB = np.diag([10.0, 10.0, 5.0, 5.0, 1e-4]) 
GROUP_ASSOCIATION_THRESH = 150.0 

# ======================================================
# Helper Classes & Functions
# ======================================================
class GroupKalmanFilter:
    """A simple KF for tracking group centroids to stabilize ID association."""
    def __init__(self, pos):
        self.x = np.array([pos[0], pos[1], 0, 0])
        self.P = np.eye(4) * 50.0
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.eye(2) * 25.0 # Assume high measurement noise for centroids
        self.Q = np.eye(4) * 4.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

def generate_sigma_points(mean, cov):
    points = np.zeros((2 * N_DIM + 1, N_DIM))
    try:
        L = cholesky((N_DIM + LAMBDA) * (cov + np.eye(N_DIM) * 1e-6))
    except np.linalg.LinAlgError:
        L = np.sqrt(N_DIM + LAMBDA) * np.sqrt(np.abs(np.diag(cov))) * np.eye(N_DIM)
    points[0] = mean
    for i in range(N_DIM):
        points[i + 1] = mean + L[i, :]; points[N_DIM + 1 + i] = mean - L[i, :]
    return points

def get_weights():
    wm = np.full(2*N_DIM+1, 0.5/(N_DIM+LAMBDA)); wc = wm.copy()
    wm[0] = LAMBDA/(N_DIM+LAMBDA); wc[0] = wm[0] + (1 - ALPHA**2 + BETA)
    return wm, wc

def rotate_vector(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([vec[0]*c - vec[1]*s, vec[0]*s + vec[1]*c])

# ======================================================
class LabeledGaussianComponent:
    def __init__(self, mean, cov, r, label):
        self.m, self.P, self.r, self.label = mean, cov, r, label
        self.miss_streak = 0 

class GraphMBTracker:
    def __init__(self):
        self.p_survival, self.p_detect = 0.99, 0.98
        self.clutter_density = 1e-5; self.dt = 1.0 
        self.confirm_thresh = 0.70 # 回调到更稳健的阈值
        self.prune_thresh = 0.01; self.max_components = 100
        self.tracks = []; self.next_point_id = 1
        self.next_group_id = 1; self.active_groups = {}
        self.wm, self.wc = get_weights()
        self.cluster_model = DBSCAN(eps=35, min_samples=2)

    def reset(self):
        self.tracks, self.active_groups = [], {}
        self.next_point_id, self.next_group_id = 1, 1

    def _find_parent(self, child_idx, all_tracks, track_positions, track_velocities):
        child_pos = track_positions[child_idx]
        dists = euclidean_distances(child_pos.reshape(1, -1), track_positions)[0]
        dists[child_idx] = np.inf
        
        candidate_indices = np.where((dists > 0) & (dists < PARENT_SEARCH_RADIUS))[0]
        if not candidate_indices.any(): return None, None
        
        best_parent_idx = candidate_indices[np.argmin(dists[candidate_indices])]
        parent_track = all_tracks[best_parent_idx]
        displacement = all_tracks[child_idx].m[:2] - parent_track.m[:2]
        return parent_track, displacement

    def _motion_model(self, x_state):
        x, y, vx, vy, w = x_state; T = self.dt
        if abs(w) < 1e-5: return np.array([x+vx*T, y+vy*T, vx, vy, w])
        sin_wt, cos_wt = np.sin(w*T), np.cos(w*T)
        nx = x + (vx/w)*sin_wt - (vy/w)*(1-cos_wt)
        ny = y + (vy/w)*(1-cos_wt) + (vx/w)*sin_wt
        nvx, nvy = vx*cos_wt - vy*sin_wt, vx*sin_wt + vy*cos_wt
        return np.array([nx, ny, nvx, nvy, w])
    
    def _measurement_model(self, x_state): return x_state[:2]
        
    def step(self, measurements):
        # Parts 1, 2, 3: Core UKF point tracking (no changes)
        if len(measurements) == 0: measurements = np.empty((0, 2))

        if self.tracks:
            track_positions = np.array([t.m[:2] for t in self.tracks])
            track_velocities = np.array([t.m[2:4] for t in self.tracks]) 
            parent_info = [self._find_parent(i, self.tracks, track_positions, track_velocities) for i in range(len(self.tracks))]
        else: parent_info = []

        for i, trk in enumerate(self.tracks):
            trk.r *= self.p_survival
            parent, displacement = parent_info[i]
            if parent:
                parent_w = parent.m[4]
                rotated_displacement = rotate_vector(displacement, parent_w * self.dt)
                sigma_points = generate_sigma_points(parent.m, parent.P)
                propagated_points = np.array([self._motion_model(s) for s in sigma_points])
                propagated_points[:, 0] += rotated_displacement[0] 
                propagated_points[:, 1] += rotated_displacement[1] 
                Q_total = Q_PROCESS_BASE + Q_COLLAB
            else:
                sigma_points = generate_sigma_points(trk.m, trk.P)
                propagated_points = np.array([self._motion_model(s) for s in sigma_points])
                Q_total = Q_PROCESS_BASE
            pred_m = np.sum(self.wm[:, np.newaxis] * propagated_points, axis=0)
            pred_P = np.zeros((N_DIM, N_DIM))
            for k in range(2 * N_DIM + 1):
                diff = propagated_points[k] - pred_m
                pred_P += self.wc[k] * np.outer(diff, diff)
            pred_P += Q_total
            trk.m, trk.P = pred_m, (pred_P + pred_P.T) / 2.0

        num_tracks, num_meas = len(self.tracks), len(measurements)
        R_MATRIX = np.eye(2) * 3.0
        track_predictions = {}
        for t_idx, trk in enumerate(self.tracks):
            sigma_points = generate_sigma_points(trk.m, trk.P)
            meas_points = np.array([self._measurement_model(s) for s in sigma_points])
            z_pred = np.sum(self.wm[:, np.newaxis] * meas_points, axis=0)
            S = np.zeros((2, 2))
            for k in range(2 * N_DIM + 1): S += self.wc[k] * np.outer(meas_points[k] - z_pred, meas_points[k] - z_pred)
            S += R_MATRIX; S = (S + S.T) / 2.0
            P_xz = np.zeros((N_DIM, 2))
            for k in range(2 * N_DIM + 1): P_xz += self.wc[k] * np.outer(sigma_points[k] - trk.m, meas_points[k] - z_pred)
            track_predictions[t_idx] = (z_pred, S, P_xz)

        cost_matrix = np.full((num_tracks, num_meas), 1000.0)
        for t_idx in range(num_tracks):
            z_pred, S, _ = track_predictions[t_idx]
            try:
                S_inv, det_S = np.linalg.inv(S), np.linalg.det(S)
                if det_S <= 1e-20: continue
            except np.linalg.LinAlgError: continue
            for m_idx, z in enumerate(measurements):
                diff = z - z_pred; mahal_dist_sq = diff.T @ S_inv @ diff
                if mahal_dist_sq < 16.0: cost_matrix[t_idx, m_idx] = 0.5 * (mahal_dist_sq + np.log(det_S) + 2 * np.log(2 * np.pi))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_tracks, assigned_meas = set(), set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > 20.0: continue
            trk = self.tracks[r]; z = measurements[c]; z_pred, S, P_xz = track_predictions[r]
            K = P_xz @ np.linalg.inv(S); trk.m += K @ (z - z_pred); trk.P -= K @ S @ K.T
            trk.P = (trk.P + trk.P.T) / 2.0
            likelihood = np.exp(-cost_matrix[r, c])
            num = trk.r * self.p_detect * likelihood
            den = (1 - trk.r * self.p_detect) * self.clutter_density + num
            trk.r = num / (den + 1e-9); trk.miss_streak = 0
            assigned_tracks.add(r); assigned_meas.add(c)

        for i in range(num_tracks):
            if i not in assigned_tracks: trk = self.tracks[i]; trk.r *= (1 - self.p_detect) / (1 - trk.r * self.p_detect + 1e-9); trk.miss_streak += 1
        for i in range(num_meas):
            if i not in assigned_meas:
                m_init = np.array([measurements[i][0], measurements[i][1], 0, 0, 0])
                P_init = np.diag([25., 25., 100., 100., 0.5])
                self.tracks.append(LabeledGaussianComponent(m_init, P_init, 0.1, self.next_point_id)); self.next_point_id += 1
        
        self.tracks = [t for t in self.tracks if t.r > self.prune_thresh and t.miss_streak < 5]
        if len(self.tracks) > self.max_components: self.tracks.sort(key=lambda x: x.r, reverse=True); self.tracks = self.tracks[:self.max_components]

        # ======================================================
        # 4. 后处理：DBSCAN 聚类 + 带KF的ID关联
        # ======================================================
        confirmed_tracks = [t for t in self.tracks if t.r > self.confirm_thresh]
        point_labels = np.full(len(measurements), -1, dtype=int)
        if not confirmed_tracks: return np.array([]), np.array([]), point_labels

        positions = np.array([t.m[:2] for t in confirmed_tracks])
        labels = self.cluster_model.fit_predict(positions)
        
        current_clusters = []
        unique_labels = set(labels) - {-1}
        for label in unique_labels:
            member_indices = np.where(labels == label)[0]
            center = np.mean(positions[member_indices], axis=0)
            current_clusters.append({'center': center, 'comp_id': label})
        
        if not current_clusters: return np.array([]), np.array([]), point_labels

        # [CRITICAL UPGRADE] 使用KF预测群组位置
        active_gids = list(self.active_groups.keys())
        cost_matrix_g = np.full((len(active_gids), len(current_clusters)), 1000.0)
        
        for r, gid in enumerate(active_gids):
            pred_center = self.active_groups[gid]['kf'].predict() # Predict!
            for c, cluster in enumerate(current_clusters):
                dist = np.linalg.norm(pred_center - cluster['center'])
                if dist < GROUP_ASSOCIATION_THRESH: cost_matrix_g[r, c] = dist
                    
        row_g, col_g = linear_sum_assignment(cost_matrix_g)
        final_centers, final_ids, comp_to_gid = [], [], {}
        matched_cluster_indices, matched_groups = set(), set()
        
        for r, c in zip(row_g, col_g):
            if cost_matrix_g[r, c] < GROUP_ASSOCIATION_THRESH:
                gid = active_gids[r]; cluster = current_clusters[c]
                self.active_groups[gid]['kf'].update(cluster['center']) # Update!
                self.active_groups[gid]['age'] = 0
                final_centers.append(cluster['center']); final_ids.append(gid)
                comp_to_gid[cluster['comp_id']] = gid
                matched_groups.add(gid); matched_cluster_indices.add(c)
                
        for c in range(len(current_clusters)):
            if c not in matched_cluster_indices:
                gid = self.next_group_id; self.next_group_id += 1
                cluster = current_clusters[c]
                self.active_groups[gid] = {'kf': GroupKalmanFilter(cluster['center']), 'age': 0} # New KF!
                final_centers.append(cluster['center']); final_ids.append(gid)
                comp_to_gid[cluster['comp_id']] = gid
                
        dead_gids = [gid for gid in self.active_groups if gid not in matched_groups]
        for gid in dead_gids: self.active_groups[gid]['age'] += 1
        self.active_groups = {g: t for g, t in self.active_groups.items() if t['age'] <= 5}

        if len(measurements) > 0:
            dists_m_t = euclidean_distances(measurements, positions)
            closest_track_idx = np.argmin(dists_m_t, axis=1)
            min_dists = np.min(dists_m_t, axis=1)
            for m_i, t_i in enumerate(closest_track_idx):
                if min_dists[m_i] < 50.0:
                    comp_label = labels[t_i]
                    if comp_label in comp_to_gid: point_labels[m_i] = comp_to_gid[comp_label]
        
        return np.array(final_centers), np.array(final_ids), point_labels