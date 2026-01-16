import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

class LabeledGaussianComponent:
    def __init__(self, mean, cov, r, label):
        self.m = mean        # State: [x, y, vx, vy]
        self.P = cov         # Covariance
        self.r = r           # Existence Probability
        self.label = label   # Track ID
        self.miss_streak = 0 # For pruning

class CBMeMBerTracker:
    """
    Labeled GM-CBMeMBer (Approximation) - Fixed for Infeasible Matrix
    """
    def __init__(self):
        # --- Model Parameters ---
        self.p_survival = 0.99
        self.p_detect = 0.95
        self.clutter_density = 1e-6 
        
        # --- Birth Parameters ---
        self.birth_prob_init = 0.1 
        
        # --- Thresholds ---
        self.confirm_thresh = 0.8  
        self.prune_thresh = 0.05   
        self.gating_dist = 40.0    
        
        # --- State ---
        self.tracks = [] 
        self.next_id = 1

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def step(self, measurements):
        # 1. Prediction
        F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        Q = np.eye(4) * 2.0 
        
        for trk in self.tracks:
            trk.m = F @ trk.m
            trk.P = F @ trk.P @ F.T + Q
            trk.r = self.p_survival * trk.r
        
        # 2. Gating & Association
        num_tracks = len(self.tracks)
        num_meas = len(measurements)
        
        # FIX: 使用大有限数代替 np.inf，防止 linear_sum_assignment 崩溃
        LARGE_COST = 10000.0 
        cost_matrix = np.full((num_tracks, num_meas), LARGE_COST)
        
        H = np.array([[1,0,0,0],[0,1,0,0]])
        R = np.eye(2) * 5.0
        
        for t_idx, trk in enumerate(self.tracks):
            z_pred = H @ trk.m
            S = H @ trk.P @ H.T + R
            inv_S = np.linalg.inv(S)
            
            for m_idx, z in enumerate(measurements):
                diff = z - z_pred
                mahalanobis = np.sqrt(diff.T @ inv_S @ diff)
                
                if mahalanobis < 5.0: # Gating
                    try:
                        likelihood = multivariate_normal.pdf(z, mean=z_pred, cov=S)
                    except:
                        likelihood = 1e-10
                    
                    # Cost = Negative Log Likelihood
                    # likelihood 越小 cost 越大
                    val = -np.log(likelihood + 1e-10)
                    cost_matrix[t_idx, m_idx] = val

        # Linear Assignment
        # 如果矩阵为空（没有track或没有measure），跳过
        if num_tracks > 0 and num_meas > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = [], []
        
        assigned_tracks = set()
        assigned_meas = set()
        
        # 3. Update Matched Tracks
        for r, c in zip(row_ind, col_ind):
            # FIX: 检查 cost 是否有效 (不是初始的 LARGE_COST)
            if cost_matrix[r, c] >= LARGE_COST - 1.0: 
                continue
            
            trk = self.tracks[r]
            z = measurements[c]
            
            # Kalman Update
            y = z - H @ trk.m
            S = H @ trk.P @ H.T + R
            K = trk.P @ H.T @ np.linalg.inv(S)
            
            trk.m = trk.m + K @ y
            trk.P = (np.eye(4) - K @ H) @ trk.P
            
            # Probability Update
            likelihood = np.exp(-cost_matrix[r, c])
            
            numerator = trk.r * self.p_detect * likelihood
            denominator = (1 - trk.r * self.p_detect) * self.clutter_density + numerator
            
            new_r = numerator / (denominator + 1e-10)
            trk.r = min(0.999, new_r + 0.1) 
            trk.miss_streak = 0
            
            assigned_tracks.add(r)
            assigned_meas.add(c)
            
        # 4. Update Unmatched Tracks (Missed Detection)
        for i in range(num_tracks):
            if i not in assigned_tracks:
                trk = self.tracks[i]
                numerator = trk.r * (1 - self.p_detect)
                denominator = 1 - trk.r * self.p_detect
                trk.r = numerator / (denominator + 1e-10)
                trk.miss_streak += 1

        # 5. Birth (New Tracks)
        for i in range(num_meas):
            if i not in assigned_meas:
                z = measurements[i]
                new_track = LabeledGaussianComponent(
                    mean=np.array([z[0], z[1], 0, 0]),
                    cov=np.eye(4) * 50.0,
                    r=self.birth_prob_init, 
                    label=self.next_id
                )
                self.next_id += 1
                self.tracks.append(new_track)
                
        # 6. Pruning & Extraction
        final_tracks = []
        ret_c, ret_id = [], []
        
        for trk in self.tracks:
            # Extraction
            if trk.r > self.confirm_thresh:
                ret_c.append(trk.m[:2])
                ret_id.append(trk.label)
                final_tracks.append(trk)
            # Pruning
            elif trk.r > self.prune_thresh and trk.miss_streak < 3:
                final_tracks.append(trk)
            
        self.tracks = final_tracks
        
        return np.array(ret_c), np.array(ret_id)