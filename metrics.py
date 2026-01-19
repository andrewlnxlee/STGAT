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
        # --- 基础计数 ---
        self.total_frames = 0
        self.total_time_sec = 0.0
        
        # --- MOTA / MOTP 相关 ---
        self.total_misses = 0
        self.total_fps = 0  # False Positives
        self.total_id_switches = 0
        self.total_gt_objects = 0
        self.total_dist_error = 0.0
        self.total_matches = 0
        
        # --- OSPA 相关 ---
        self.total_ospa = 0.0
        self.total_ospa_loc = 0.0 
        self.total_ospa_card = 0.0
        
        # --- 聚类与数量误差 ---
        self.centroid_mse = 0.0
        self.cardinality_error = 0.0
        self.cardinality_samples = 0
        self.total_purity_score = 0.0
        self.total_completeness_score = 0.0
        self.total_points = 0
        
        # --- 辅助 ---
        self.gt_id_map = {}

        # --- 新增: G-IoU 相关 ---
        self.total_iou = 0.0
        self.iou_matches = 0  # 确保变量名一致

    def update_time(self, seconds):
        self.total_time_sec += seconds

    def update_clustering_metrics(self, point_gt_labels, point_pred_labels):
        if len(point_gt_labels) == 0: return
        valid_mask = (point_gt_labels > 0) & (point_pred_labels > -1)
        gt_labels, pred_labels = point_gt_labels[valid_mask], point_pred_labels[valid_mask]
        if len(gt_labels) == 0: return
        self.total_points += len(gt_labels)

        # Purity
        for pred_id in np.unique(pred_labels):
            mask = pred_labels == pred_id
            gt_in_cluster = gt_labels[mask]
            if len(gt_in_cluster) > 0:
                self.total_purity_score += Counter(gt_in_cluster).most_common(1)[0][1]
        
        # Completeness
        for gt_id in np.unique(gt_labels):
            mask = gt_labels == gt_id
            pred_for_gt = pred_labels[mask]
            if len(pred_for_gt) > 0:
                self.total_completeness_score += Counter(pred_for_gt).most_common(1)[0][1]
    
    # 计算 IoU
    def _compute_iou(self, box1, box2):
        # box: [cx, cy, w, h]
        b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
        b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
        b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
        b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        b1_area = box1[2] * box1[3]
        b2_area = box2[2] * box2[3]
        
        union_area = b1_area + b2_area - inter_area
        if union_area <= 1e-6: return 0.0
        return inter_area / union_area

    # 更新函数
    def update(self, gt_centers, gt_ids, pred_centers, pred_ids, gt_shapes=None, pred_shapes=None):
        self.total_frames += 1
        
        # 格式化
        if not isinstance(gt_centers, np.ndarray) or gt_centers.ndim != 2: 
            gt_centers = np.array(gt_centers).reshape(-1, 2)
        if not isinstance(pred_centers, np.ndarray) or pred_centers.ndim != 2: 
            pred_centers = np.array(pred_centers).reshape(-1, 2)
        
        num_gt = gt_centers.shape[0]
        num_pred = pred_centers.shape[0]
        
        # 1. OSPA 计算
        ospa, ospa_loc, ospa_card = self._compute_ospa_decomposed(gt_centers, pred_centers)
        self.total_ospa += ospa
        self.total_ospa_loc += ospa_loc
        self.total_ospa_card += ospa_card
        
        self.total_gt_objects += num_gt
        self.cardinality_error += abs(num_gt - num_pred)
        self.cardinality_samples += 1
        
        if num_gt == 0: 
            self.total_fps += num_pred
            return
        if num_pred == 0: 
            self.total_misses += num_gt
            return 
        
        # 2. MOTA 匹配
        dist_matrix = euclidean_distances(gt_centers, pred_centers)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < 40.0:
                matches.append((r, c))
        
        self.total_matches += len(matches)
        self.total_misses += (num_gt - len(matches))
        self.total_fps += (num_pred - len(matches))
        
        for r, c in matches:
            dist = dist_matrix[r, c]
            self.total_dist_error += dist
            self.centroid_mse += dist**2
            
            # ID Switch
            gt_id = gt_ids[r]
            track_id = pred_ids[c]
            
            if gt_id in self.gt_id_map:
                if self.gt_id_map[gt_id] != track_id:
                    self.total_id_switches += 1
            self.gt_id_map[gt_id] = track_id
            
            # 3. G-IoU 计算
            if gt_shapes is not None and pred_shapes is not None:
                if r < len(gt_shapes) and c < len(pred_shapes):
                    gt_box = [gt_centers[r][0], gt_centers[r][1], gt_shapes[r][0], gt_shapes[r][1]]
                    pred_box = [pred_centers[c][0], pred_centers[c][1], pred_shapes[c][0], pred_shapes[c][1]]
                    
                    self.total_iou += self._compute_iou(gt_box, pred_box)
                    self.iou_matches += 1

    def _compute_ospa_decomposed(self, gt, pred):
        m, n = gt.shape[0], pred.shape[0]
        if m == 0 and n == 0: return 0.0, 0.0, 0.0
        if m == 0 or n == 0: return self.ospa_c, 0.0, self.ospa_c
        
        dist_mat = np.minimum(euclidean_distances(gt, pred), self.ospa_c)
        row, col = linear_sum_assignment(dist_mat)
        
        matched_sum = np.sum(dist_mat[row, col] ** self.ospa_p)
        card_penalty = abs(n - m) * (self.ospa_c ** self.ospa_p)
        
        total = (matched_sum + card_penalty) / max(m, n)
        
        return total**(1/self.ospa_p), (matched_sum/max(m,n))**(1/self.ospa_p), (card_penalty/max(m,n))**(1/self.ospa_p)

    def compute(self):
        frames = max(1, self.total_frames)
        gt_objs = max(1, self.total_gt_objects)
        matches = max(1, self.total_matches)
        points = max(1, self.total_points)
        
        # 计算所有指标
        mota = 1.0 - (self.total_misses + self.total_fps + self.total_id_switches) / gt_objs
        motp = self.total_dist_error / matches if matches > 0 else 0.0
        
        return {
            "MOTA": mota,
            "MOTP": motp,
            "OSPA (Total)": self.total_ospa / frames,
            "OSPA (Loc)": self.total_ospa_loc / frames,
            "OSPA (Card)": self.total_ospa_card / frames,
            "RMSE (Pos)": np.sqrt(self.centroid_mse / matches) if matches > 0 else 0.0,
            "IDSW": self.total_id_switches,
            "FAR": self.total_fps / frames,
            "Count Err": self.cardinality_error / max(1, self.cardinality_samples),
            "G-IoU": self.total_iou / max(1, self.iou_matches),
            "Purity": self.total_purity_score / points,
            "Comp": self.total_completeness_score / points,
            "Time": (self.total_time_sec / frames) * 1000
        }