import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

class LabeledGaussianComponent:
    def __init__(self, mean, cov, r, label):
        self.m = mean        # State: [x, y, vx, vy, omega]
        self.P = cov         # Covariance (5x5)
        self.r = r           # Existence Probability
        self.label = label   # Track ID
        self.miss_streak = 0 

class GraphMBTracker:
    """
    Graph-based Tracker with EKF-CT Model.
    (Simplified GNN association version of CBMeMBer for efficiency)
    """
    def __init__(self):
        # --- Parameters ---
        self.p_survival = 0.99
        self.p_detect = 0.98
        self.clutter_density = 1e-6 
        self.dt = 1.0 
        
        # Thresholds
        self.confirm_thresh = 0.7 
        self.prune_thresh = 0.01   
        self.max_components = 100   
        self.graph_dist_thresh = 150.0  # Slightly increased for stability
        
        # State
        self.tracks = [] 
        self.next_point_id = 1
        
        # Group Level
        self.group_tracks = {} 
        self.next_group_id = 1
        self.max_group_age = 10 # More robust group retention

    def reset(self):
        self.tracks = []
        self.next_point_id = 1
        self.group_tracks = {}
        self.next_group_id = 1

    def step(self, measurements):
        """
        measurements: (N, 2) np.array
        """
        if len(measurements) == 0:
            measurements = np.empty((0, 2))

        # ======================================================
        # Part 1: EKF Prediction (Corrected CT Model)
        # ======================================================
        
        # Process Noise: increase omega noise to allow learning turn rate
        Q = np.diag([1.0, 1.0, 4.0, 4.0, 1e-4]) 
        
        for trk in self.tracks:
            x, y, vx, vy, w = trk.m
            T = self.dt
            
            # --- State Transition ---
            if abs(w) < 1e-5: # CV Model (Linear)
                nx = x + vx * T
                ny = y + vy * T
                nvx = vx
                nvy = vy
                nw = w
                
                F = np.eye(5)
                F[0, 2] = T
                F[1, 3] = T
            else: # CT Model (Nonlinear)
                sin_wt = np.sin(w * T)
                cos_wt = np.cos(w * T)
                
                # 1. State Prediction
                nx = x + (vx/w)*sin_wt - (vy/w)*(1 - cos_wt)
                ny = y + (vx/w)*(1 - cos_wt) + (vy/w)*sin_wt
                nvx = vx*cos_wt - vy*sin_wt
                nvy = vx*sin_wt + vy*cos_wt
                nw = w
                
                # 2. Jacobian Calculation (CRITICAL FIX)
                F = np.eye(5)
                
                # Partial derivatives w.r.t vx, vy
                F[0, 2] = sin_wt/w
                F[0, 3] = -(1 - cos_wt)/w
                F[1, 2] = (1 - cos_wt)/w
                F[1, 3] = sin_wt/w
                F[2, 2] = cos_wt
                F[2, 3] = -sin_wt
                F[3, 2] = sin_wt
                F[3, 3] = cos_wt
                
                # Partial derivatives w.r.t omega (The missing part!)
                # Derived from limit calculus or wolfram alpha
                # term1 = T*w*cos(wT) - sin(wT)
                term1 = T * w * cos_wt - sin_wt
                # term2 = T*w*sin(wT) - 1 + cos(wT)
                term2 = T * w * sin_wt - 1 + cos_wt
                
                F[0, 4] = (vx * term1 - vy * term2) / (w**2) # dx/dw
                F[1, 4] = (vx * term2 + vy * term1) / (w**2) # dy/dw
                
                F[2, 4] = -T * vx * sin_wt - T * vy * cos_wt # dvx/dw
                F[3, 4] =  T * vx * cos_wt - T * vy * sin_wt # dvy/dw
                F[4, 4] = 1.0 # dw/dw
                
            trk.m = np.array([nx, ny, nvx, nvy, nw])
            trk.P = F @ trk.P @ F.T + Q
            trk.r = self.p_survival * trk.r
            
        # ======================================================
        # Part 2: Gating & Hard Association (GNN)
        # ======================================================
        num_tracks = len(self.tracks)
        num_meas = len(measurements)
        
        # Cost matrix: High value for initialization
        cost_matrix = np.full((num_tracks, num_meas), 1000.0)
        
        H = np.zeros((2, 5))
        H[0, 0] = 1; H[1, 1] = 1
        R = np.eye(2) * 25.0 # Increase measurement noise slightly for robustness
        
        if num_tracks > 0 and num_meas > 0:
            track_pos = np.array([t.m[:2] for t in self.tracks])
            dists = euclidean_distances(track_pos, measurements)
            
            # Coarse Gating
            possible_pairs = np.argwhere(dists < 60.0)
            
            for t_idx, m_idx in possible_pairs:
                trk = self.tracks[t_idx]
                z = measurements[m_idx]
                
                # EKF Update Variables
                z_pred = H @ trk.m
                S = H @ trk.P @ H.T + R
                
                # Mahalanobis Distance
                diff = z - z_pred
                try:
                    S_inv = np.linalg.inv(S)
                    mahal = diff.T @ S_inv @ diff
                    det_S = np.linalg.det(S)
                    likelihood = np.exp(-0.5 * mahal) / (2 * np.pi * np.sqrt(det_S))
                except:
                    likelihood = 1e-20
                
                if likelihood > 1e-20:
                    cost_matrix[t_idx, m_idx] = -np.log(likelihood)

        # Hungarian Assignment
        row_ind, col_ind = [], []
        if num_tracks > 0 and num_meas > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
        assigned_tracks = set()
        assigned_meas = set()
        
        # --- Update Assigned Tracks ---
        for r, c in zip(row_ind, col_ind):
            # Gating Threshold for assignment cost
            if cost_matrix[r, c] > 20.0: continue
            
            trk = self.tracks[r]
            z = measurements[c]
            
            # EKF Update
            z_pred = H @ trk.m
            S = H @ trk.P @ H.T + R
            K = trk.P @ H.T @ np.linalg.inv(S)
            trk.m = trk.m + K @ (z - z_pred)
            trk.P = (np.eye(5) - K @ H) @ trk.P
            
            # Existence Probability Update (Simplified Bayesian)
            likelihood = np.exp(-cost_matrix[r, c])
            # Numerator: P(Detection | Exists) * P(Exists)
            num = likelihood * self.p_detect * trk.r
            # Denominator: P(Detection) = P(Det|Exist)*P(Exist) + P(FalseAlarm)
            den = num + self.clutter_density * (1 - trk.r) # Simplified term
            
            trk.r = num / (den + 1e-20)
            trk.r = min(0.999, trk.r + 0.1) # Bonus for detection
            trk.miss_streak = 0
            
            assigned_tracks.add(r)
            assigned_meas.add(c)
            
        # --- Handle Missed Tracks ---
        for i in range(num_tracks):
            if i not in assigned_tracks:
                trk = self.tracks[i]
                # P(Exists | No Detection)
                num = trk.r * (1 - self.p_detect)
                den = num + (1 - trk.r) # Simplified
                trk.r = num / (den + 1e-10)
                trk.miss_streak += 1
                
        # --- Handle Births ---
        birth_tracks = []
        for i in range(num_meas):
            if i not in assigned_meas:
                z = measurements[i]
                # Init: [x, y, 0, 0, 0]
                # High cov on velocity (100) and omega (0.5)
                m_init = np.array([z[0], z[1], 0, 0, 0])
                P_init = np.diag([10.0, 10.0, 100.0, 100.0, 0.5])
                
                birth_tracks.append(LabeledGaussianComponent(
                    mean=m_init, cov=P_init,
                    r=0.1, label=self.next_point_id # Low prob init
                ))
                self.next_point_id += 1
        
        self.tracks.extend(birth_tracks)
        
        # --- Pruning ---
        # Sort by existence probability
        self.tracks.sort(key=lambda x: x.r, reverse=True)
        # Keep confirmed tracks or recent births
        self.tracks = [t for t in self.tracks if t.r > self.prune_thresh and t.miss_streak < 5]
        if len(self.tracks) > self.max_components:
            self.tracks = self.tracks[:self.max_components]
        
        # ======================================================
        # Part 3: Graph Construction (Logic is fine)
        # ======================================================
        confirmed_tracks = [t for t in self.tracks if t.r > self.confirm_thresh]
        point_labels = np.full(len(measurements), -1)

        if not confirmed_tracks:
            return np.array([]), np.array([]), point_labels

        positions = np.array([t.m[:2] for t in confirmed_tracks])
        
        # Distance based graph
        dist_mat = euclidean_distances(positions)
        adj = (dist_mat < self.graph_dist_thresh).astype(int)
        np.fill_diagonal(adj, 0)
        
        n_components, labels = connected_components(coo_matrix(adj), directed=False)
        
        current_centers = []
        # Mapping: current_group_idx -> [track_indices in confirmed_tracks]
        curr_group_members = {} 
        
        for i in range(n_components):
            member_indices = np.where(labels == i)[0]
            if len(member_indices) < 2: continue # Ignore singletons
            
            center = np.mean(positions[member_indices], axis=0)
            current_centers.append(center)
            curr_group_members[len(current_centers)-1] = member_indices
            
        current_centers = np.array(current_centers)
        
        # ======================================================
        # Part 4: Group ID Association (Linear KF)
        # ======================================================
        for gid, grp in self.group_tracks.items():
            grp['kf'].predict()
            grp['age'] += 1
            
        active_gids = list(self.group_tracks.keys())
        assignments = {}
        
        if len(active_gids) > 0 and len(current_centers) > 0:
            # Cost: distance between predicted group center and current detected center
            cost = np.zeros((len(active_gids), len(current_centers)))
            for i, gid in enumerate(active_gids):
                pred = self.group_tracks[gid]['kf'].x[:2]
                cost[i, :] = np.linalg.norm(current_centers - pred, axis=1)
            
            r_idx, c_idx = linear_sum_assignment(cost)
            for r, c in zip(r_idx, c_idx):
                if cost[r, c] < 200.0: # Group gating
                    gid = active_gids[r]
                    assignments[c] = gid
                    self.group_tracks[gid]['kf'].update(current_centers[c])
                    self.group_tracks[gid]['age'] = 0
                    
        # New Groups
        for c in range(len(current_centers)):
            if c not in assignments:
                gid = self.next_group_id
                self.next_group_id += 1
                self.group_tracks[gid] = {
                    'kf': GroupKalmanFilter(current_centers[c]),
                    'age': 0
                }
                assignments[c] = gid
                
        # Delete Dead Groups
        self.group_tracks = {g: t for g, t in self.group_tracks.items() if t['age'] <= self.max_group_age}
        
        # Output formatting
        final_centers = []
        final_ids = []
        
        # Assign IDs to measurements for visualization
        meas_pos = measurements
        if len(meas_pos) > 0 and len(confirmed_tracks) > 0:
             # Find which track each measurement belongs to (reuse association if possible, but here we do NN)
             # Simplification: Assign measurement to the Group ID of the closest Confirmed Track
             track_pos = np.array([t.m[:2] for t in confirmed_tracks])
             dists_m_t = euclidean_distances(meas_pos, track_pos)
             closest_track_idx = np.argmin(dists_m_t, axis=1)
             min_dists = np.min(dists_m_t, axis=1)
             
             for m_i, t_i in enumerate(closest_track_idx):
                 if min_dists[m_i] < 30.0:
                     # Find which group this track t_i belongs to
                     # t_i is index in confirmed_tracks
                     # Check labels from connected_components
                     comp_label = labels[t_i]
                     # Find which `c` (current_center index) corresponds to this comp_label
                     # We need reverse mapping from comp_label -> c
                     # Re-loop to find c
                     found_c = -1
                     for c_idx, members in curr_group_members.items():
                         if t_i in members:
                             found_c = c_idx
                             break
                     
                     if found_c != -1 and found_c in assignments:
                         point_labels[m_i] = assignments[found_c]

        for c, gid in assignments.items():
            final_centers.append(current_centers[c])
            final_ids.append(gid)
            
        return np.array(final_centers), np.array(final_ids), point_labels

class GroupKalmanFilter:
    def __init__(self, pos):
        self.x = np.array([pos[0], pos[1], 0, 0])
        self.P = np.eye(4) * 50.0
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.eye(2) * 10.0
        self.Q = np.eye(4) * 1.0
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]
        
    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try: K = self.P @ self.H.T @ np.linalg.inv(S)
        except: K = np.zeros((4,2))
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P