import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

class TrackingMetrics:
    def __init__(self, ospa_c=50.0, ospa_p=2):
        self.ospa_c = ospa_c
        self.ospa_p = ospa_p
        self.reset()

    def reset(self):
        # 基础计数
        self.total_frames = 0
        self.total_time_sec = 0.0
        
        # MOTA / MOTP 相关
        self.total_misses = 0
        self.total_fps = 0 # False Positives (虚警)
        self.total_id_switches = 0
        self.total_gt_objects = 0
        self.total_dist_error = 0.0
        self.total_matches = 0
        
        # OSPA 系列
        self.total_ospa = 0.0
        self.total_ospa_loc = 0.0  # 位置分量
        self.total_ospa_card = 0.0 # 势分量
        
        # 群目标特有
        self.centroid_mse = 0.0
        self.cardinality_error = 0.0 # 绝对数量误差
        self.cardinality_samples = 0
        self.total_purity_score = 0.0
        self.total_completeness_score = 0.0
        self.total_points = 0
        
        # 辅助变量
        self.gt_id_map = {} # 用于计算 ID Switch

    def update_time(self, seconds):
        self.total_time_sec += seconds

    def update_clustering_metrics(self, point_gt_labels, point_pred_labels):
        if len(point_gt_labels) == 0: return
        valid_mask = (point_gt_labels > 0) & (point_pred_labels > -1)
        gt_labels, pred_labels = point_gt_labels[valid_mask], point_pred_labels[valid_mask]
        if len(gt_labels) == 0: return
        self.total_points += len(gt_labels)

        # Purity (纯净度)
        for pred_id in np.unique(pred_labels):
            mask = pred_labels == pred_id; gt_in_cluster = gt_labels[mask]
            if len(gt_in_cluster) > 0: self.total_purity_score += Counter(gt_in_cluster).most_common(1)[0][1]
        # Completeness (完整度)
        for gt_id in np.unique(gt_labels):
            mask = gt_labels == gt_id; pred_for_gt = pred_labels[mask]
            if len(pred_for_gt) > 0: self.total_completeness_score += Counter(pred_for_gt).most_common(1)[0][1]
    
    def update(self, gt_centers, gt_ids, pred_centers, pred_ids):
        self.total_frames += 1
        
        # 格式标准化: 确保是 (N, 2) 的 numpy 数组
        if not isinstance(gt_centers, np.ndarray) or gt_centers.ndim != 2: 
            gt_centers = np.array(gt_centers).reshape(-1, 2)
        if not isinstance(pred_centers, np.ndarray) or pred_centers.ndim != 2: 
            pred_centers = np.array(pred_centers).reshape(-1, 2)
        
        num_gt = gt_centers.shape[0]
        num_pred = pred_centers.shape[0]
        
        # --- 1. OSPA 分解计算 (已处理空值情况) ---
        ospa, ospa_loc, ospa_card = self._compute_ospa_decomposed(gt_centers, pred_centers)
        self.total_ospa += ospa
        self.total_ospa_loc += ospa_loc
        self.total_ospa_card += ospa_card
        
        # --- 2. MOTA / MOTP 计算 ---
        self.total_gt_objects += num_gt
        self.cardinality_error += abs(num_gt - num_pred)
        self.cardinality_samples += 1
        
        # Case 1: 没有 GT，全是虚警
        if num_gt == 0:
            self.total_fps += num_pred 
            return
        
        # Case 2: 修复点 - 有 GT 但没有预测值 (全漏检)
        # 必须在计算 distance 之前返回，否则 sklearn 会报错
        if num_pred == 0:
            self.total_misses += num_gt
            return 
        
        # Case 3: 都有值，进行匹配
        dist_matrix = euclidean_distances(gt_centers, pred_centers)
        threshold = 40.0
        
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        matches = []
        
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < threshold:
                matches.append((r, c))
        
        # 统计基础指标
        self.total_matches += len(matches)
        self.total_misses += (num_gt - len(matches))
        self.total_fps += (num_pred - len(matches))
        
        # 统计误差和 ID Switch
        for r, c in matches:
            dist = dist_matrix[r, c]
            self.total_dist_error += dist
            self.centroid_mse += dist**2
            
            gt_id = gt_ids[r]
            track_id = pred_ids[c]
            
            if gt_id in self.gt_id_map:
                if self.gt_id_map[gt_id] != track_id:
                    self.total_id_switches += 1
            self.gt_id_map[gt_id] = track_id

    def _compute_ospa_decomposed(self, gt, pred):
        """计算 OSPA 及其位置/势分量"""
        m = gt.shape[0]
        n = pred.shape[0]
        
        if m == 0 and n == 0: return 0.0, 0.0, 0.0
        if m == 0 or n == 0: return self.ospa_c, 0.0, self.ospa_c # 全是势误差
        
        dist_mat = euclidean_distances(gt, pred)
        
        # OSPA 定义: min(dist, c)
        dist_mat = np.minimum(dist_mat, self.ospa_c)
        
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        
        # 匹配距离和 (Location Term sum)
        matched_sum = np.sum(dist_mat[row_ind, col_ind] ** self.ospa_p)
        
        # 势罚分 (Cardinality Term sum)
        card_diff = abs(n - m)
        card_penalty = card_diff * (self.ospa_c ** self.ospa_p)
        
        # 总 OSPA
        total_sum = matched_sum + card_penalty
        divisor = max(m, n)
        
        ospa_total = (total_sum / divisor) ** (1.0 / self.ospa_p)
        
        # 分解
        ospa_loc = (matched_sum / divisor) ** (1.0 / self.ospa_p)
        ospa_card = (card_penalty / divisor) ** (1.0 / self.ospa_p)
        
        return ospa_total, ospa_loc, ospa_card

    def compute(self):
        # 防止除零
        frames = max(1, self.total_frames)
        gt_objs = max(1, self.total_gt_objects)
        matches = max(1, self.total_matches)
        points = max(1, self.total_points)
        
        mota = 1.0 - (self.total_misses + self.total_fps + self.total_id_switches) / gt_objs
        motp = self.total_dist_error / matches
        
        # 平均每帧的指标
        res = {
            "MOTA": mota,
            "MOTP": motp,
            "OSPA (Total)": self.total_ospa / frames,
            "OSPA (Loc)": self.total_ospa_loc / frames,
            "OSPA (Card)": self.total_ospa_card / frames,
            "RMSE (Pos)": np.sqrt(self.centroid_mse / matches),
            "ID Switches": self.total_id_switches,
            "False Alarm Rate": self.total_fps / frames, # 平均每帧虚警数
            "Count Error": self.cardinality_error / max(1, self.cardinality_samples),
            "Purity": self.total_purity_score / points,
            "Completeness": self.total_completeness_score / points,
            "Time (ms)": (self.total_time_sec / frames) * 1000 # 毫秒/帧
        }
        return res