import pandas as pd
import numpy as np
import os
import shutil
import config
from tqdm import tqdm
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Dataset 
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

# 引入所有需要的类
from model import GNNGroupTracker
from dataset import RadarFileDataset
from metrics import TrackingMetrics
from trackers.baseline import BaselineTracker
from trackers.gm_cphd import GMCPHDTracker
from trackers.cbmember import CBMeMBerTracker
from trackers.graph_mb import GraphMBTracker 
from trackers.gnn_processor import GNNPostProcessor

# ================= 唯一需要配置的地方 =================
SEQUENCE_NAME = "MOT17-02"
DETECTOR_NAME = "SDP" 
GT_PATH = os.path.join("MOT17Labels", "train", f"{SEQUENCE_NAME}-{DETECTOR_NAME}", "gt", "gt.txt")
DET_PATH = os.path.join("MOT17Labels", "train", f"{SEQUENCE_NAME}-{DETECTOR_NAME}", "det", "det.txt")
OUTPUT_DIR = os.path.join(config.DATA_ROOT, "test_mot_public")
W_IMG, H_IMG = 1920, 1080
RADAR_SCALE = 1000.0
TOP_K_PER_FRAME = 50
# ====================================================

def convert_data():
    """数据转换函数 (保持原样)"""
    # ... (这部分逻辑是正确的，为了简洁，我把它省略了，请从你之前的脚本复制过来)
    # 确保 convert_data() 函数在这里被完整定义
    print("\n--- STEP 1: 强制重新生成数据 ---")
    if not os.path.exists(GT_PATH) or not os.path.exists(DET_PATH):
        print(f"❌ 找不到文件，请检查你的文件夹结构:\n  GT: {GT_PATH}\n  DET: {DET_PATH}")
        return False

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    gt_df = pd.read_csv(GT_PATH, header=None)
    gt_data = gt_df.values[gt_df.values[:, 7] == 1]
    
    det_df = pd.read_csv(DET_PATH, header=None)
    det_data = det_df.fillna(-999).values
    
    max_frame = int(gt_data[:, 0].max())
    episode_data = []
    
    total_dets_kept = 0
    for t in tqdm(range(1, max_frame + 1), desc="转换数据"):
        frame_info = {'meas': [], 'labels': [], 'gt_centers': []}
        
        curr_gt = gt_data[gt_data[:, 0] == t]
        gt_list = []
        for row in curr_gt:
            tid, x, y, w, h = int(row[1]), row[2], row[3], row[4], row[5]
            cx = (x + w/2) / W_IMG * RADAR_SCALE; cy = (y + h/2) / H_IMG * RADAR_SCALE
            if 0 <= cx <= RADAR_SCALE and 0 <= cy <= RADAR_SCALE:
                frame_info['gt_centers'].append([tid, cx, cy])
                gt_list.append({'id': tid, 'pos': np.array([cx, cy])})

        curr_det = det_data[det_data[:, 0] == t]
        if len(curr_det) > 0:
            sort_idx = np.argsort(curr_det[:, 6])[::-1]
            num_to_keep = min(len(curr_det), TOP_K_PER_FRAME)
            top_det = curr_det[sort_idx[:num_to_keep]]
            total_dets_kept += len(top_det)
            
            for row in top_det:
                x, y, w, h = row[2], row[3], row[4], row[5]
                mx = (x + w/2) / W_IMG * RADAR_SCALE; my = (y + h/2) / H_IMG * RADAR_SCALE
                if 0 <= mx <= RADAR_SCALE and 0 <= my <= RADAR_SCALE:
                    frame_info['meas'].append([mx, my])
                    matched_id = 0
                    if len(gt_list) > 0:
                        dists = cdist([[mx, my]], np.array([g['pos'] for g in gt_list]))[0]
                        min_idx = np.argmin(dists)
                        if dists[min_idx] < 40.0: matched_id = gt_list[min_idx]['id']
                    frame_info['labels'].append(matched_id)

        frame_info['meas'] = np.array(frame_info['meas'])
        frame_info['labels'] = np.array(frame_info['labels'])
        frame_info['gt_centers'] = np.array(frame_info['gt_centers'])
        
        episode_data.append(frame_info)

    save_path = os.path.join(OUTPUT_DIR, "sample_mot_public.npy")
    np.save(save_path, episode_data, allow_pickle=True)
    print(f"✅ 数据生成完毕。平均每帧检测数: {total_dets_kept/max_frame:.1f}")
    return True

class MOTPublicDataset(RadarFileDataset):
    def __init__(self):
        Dataset.__init__(self)
        self.root_dir = OUTPUT_DIR
        self.conn_radius = 30.0 # GNN建图半径保持不变
        self.file_list = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.npy')] if os.path.exists(self.root_dir) else []

def evaluate_data():
    print("\n--- STEP 2: 开始评估 (个体跟踪模式) ---")
    device = torch.device(config.DEVICE)
    test_set = MOTPublicDataset()
    
    if len(test_set) == 0: return

    gnn_model = GNNGroupTracker().to(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        gnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        gnn_model.eval()
    else:
        gnn_model = None

    # --- 【核心修改】为“个体跟踪”定制参数 ---
    # 1. DBSCAN: eps 设小，min_samples 设为 1。现在是“找独立的点”，而不是“找群”。
    # 2. GNNPostProcessor: 关联门限设小。
    trackers = {
        'Baseline': BaselineTracker(eps=15, min_samples=1), 
        'GM-CPHD': GMCPHDTracker(), # RFS系列对个体更敏感，暂时不改
        'CBMeMBer': CBMeMBerTracker(),
        'Graph-MB': GraphMBTracker(),
        'H-GAT-GT': GNNPostProcessor(dist_thresh=40.0) # 关联门限设小一点
    }
    metrics = {name: TrackingMetrics() for name in trackers.keys()}
    
    for episode_idx in tqdm(range(len(test_set)), desc="评估中"):
        episode_graphs = test_set.get(episode_idx)
        
        for name, trk in trackers.items():
            if hasattr(trk, 'reset'): trk.reset()
            if name == 'H-GAT-GT': trackers[name] = GNNPostProcessor(dist_thresh=40.0)

        for graph in episode_graphs:
            gt_data = graph.gt_centers.numpy()
            gt_centers = gt_data[:, 1:3] if len(gt_data)>0 else np.zeros((0,2))
            gt_ids = gt_data[:, 0].astype(int) if len(gt_data)>0 else np.zeros((0,))
            meas_points = graph.x.numpy()
            
            # --- H-GAT-GT ---
            # GNN的角色转变：不再是聚类器，而是“去噪+精定位器”
            # 它把属于真目标的点拉到一起，把杂波推开
            t0 = time.time()
            pred_c, pred_id, pt_map_ours = np.array([]), np.array([]), np.full(len(meas_points), -1)
            
            if gnn_model and len(meas_points) > 0:
                graph_dev = graph.to(device)
                with torch.no_grad():
                    out = gnn_model(graph_dev); _, offsets, _ = out
                
                corrected = meas_points + offsets.cpu().numpy()
                
                # 【重要】聚类参数适配个体
                labels = DBSCAN(eps=15, min_samples=1).fit(corrected).labels_
                
                det_c, det_indices = [], []
                for l in set(labels):
                    if l == -1: continue
                    idx = np.where(labels == l)[0]
                    det_indices.append(idx)
                    # 直接把修正后的点作为检测结果（因为现在一个点就是一个目标）
                    det_c.append(np.mean(corrected[idx], axis=0)) 
                
                det_c = np.array(det_c)
                
                if len(det_c) > 0:
                    pred_c, pred_id, _ = trackers['H-GAT-GT'].update(det_c, None)
                    cost = cdist(pred_c, det_c)
                    r, c = linear_sum_assignment(cost)
                    for ri, ci in zip(r, c):
                        if cost[ri, ci] < 40.0:
                            pt_map_ours[det_indices[ci]] = pred_id[ri]

            metrics['H-GAT-GT'].update_time(time.time() - t0)
            metrics['H-GAT-GT'].update(gt_centers, gt_ids, pred_c, pred_id)
            metrics['H-GAT-GT'].update_clustering_metrics(graph.point_labels.cpu().numpy(), pt_map_ours)

            # --- Baselines ---
            # Baseline现在也用个体参数
            bc, bid, bmap = trackers['Baseline'].step(meas_points)
            metrics['Baseline'].update(gt_centers, gt_ids, bc, bid)
            metrics['Baseline'].update_clustering_metrics(graph.point_labels.cpu().numpy(), bmap)
            
            # RFS 系列输入的是“检测”，而不是原始点云，所以需要预处理
            # 它们内部参数对个体跟踪更鲁棒，暂时不改
            base_dets = [np.mean(meas_points[DBSCAN(eps=15, min_samples=1).fit(meas_points).labels_ == l], axis=0) for l in set(DBSCAN(eps=15, min_samples=1).fit(meas_points).labels_) if l != -1]

            cc, cid = trackers['GM-CPHD'].step(base_dets)
            metrics['GM-CPHD'].update(gt_centers, gt_ids, cc, cid)

            mc, mid = trackers['CBMeMBer'].step(base_dets)
            metrics['CBMeMBer'].update(gt_centers, gt_ids, mc, mid)

            gc, gid, gmap = trackers['Graph-MB'].step(meas_points)
            metrics['Graph-MB'].update(gt_centers, gt_ids, gc, gid)
            metrics['Graph-MB'].update_clustering_metrics(graph.point_labels.cpu().numpy(), gmap)

    res = {k: v.compute() for k, v in metrics.items()}
    df = pd.DataFrame(res).T
    cols = ['MOTA', 'G-IoU', 'OSPA (Total)', 'IDSW', 'RMSE (Pos)', 'Purity', 'Time']
    print("\n" + "="*80)
    print(f"MOTChallenge {SEQUENCE_NAME}-{DETECTOR_NAME} Evaluation (Individual Tracking Mode)")
    print("="*80)
    print(df[cols].to_string())
    print("="*80)

if __name__ == "__main__":
    if convert_data():
        evaluate_data()