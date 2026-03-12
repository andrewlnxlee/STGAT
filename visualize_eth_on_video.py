import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import imageio
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import config
from model import GNNGroupTracker
from trackers.gnn_processor import GNNPostProcessor

# --- 坐标转换与校准 ---
def world_to_pixel(x, y, H_inv, u_offset=50, v_offset=-50):
    point = np.array([y, x, 1.0]).reshape(3, 1) 
    pixel = H_inv @ point
    if pixel[2, 0] == 0: return 0, 0
    u = pixel[0, 0] / pixel[2, 0]
    v = pixel[1, 0] / pixel[2, 0]
    return int(u + u_offset), int(v + v_offset)

def parse_obsmat(filepath):
    data = np.loadtxt(filepath)
    frames = defaultdict(list)
    for row in data:
        frames[int(row[0])].append({'id': int(row[1]), 'pos': [row[2], row[4]]})
    return frames

def visualize_on_video():
    video_path = "datasets/ewap_dataset/seq_eth/seq_eth.avi"
    h_matrix_path = "datasets/ewap_dataset/seq_eth/H.txt"
    obsmat_path = "datasets/ewap_dataset/seq_eth/obsmat.txt"
    model_path = config.MODEL_SAVE_PATH
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    H = np.loadtxt(h_matrix_path)
    H_inv = np.linalg.inv(H)
    obs_data = parse_obsmat(obsmat_path)
    sorted_frames = sorted(obs_data.keys())

    gnn_model = GNNGroupTracker().to(device)
    if os.path.exists(model_path):
        gnn_model.load_state_dict(torch.load(model_path, map_location=device))
        gnn_model.eval()

    tracker = GNNPostProcessor()

    try:
        reader = imageio.get_reader(video_path)
    except Exception as e:
        print(f"❌ 无法读取视频: {e}")
        return

    save_path = "eth_final_v3.mp4"
    # 使用 10fps，确保动作连贯
    writer = imageio.get_writer(save_path, fps=10, quality=9, codec='libx264')
    
    history = {} 
    colors = {} 
    
    # 增加渲染长度到 1500 点
    print(f"正在生成终极版高清演示视频 (共 {len(sorted_frames[:1500])} 采样点)...")
    for fid in tqdm(sorted_frames[:1500]):
        try:
            frame = reader.get_data(fid)
        except: continue
        
        frame = frame.copy()
        peds = obs_data[fid]
        raw_pos = np.array([p['pos'] for p in peds])
        scaled_pos = raw_pos * config.COORD_SCALE + np.array(config.COORD_OFFSET)
        
        # --- H-GAT-GT 推断 ---
        x_in = torch.tensor(scaled_pos, dtype=torch.float).to(device)
        dist_mat = cdist(scaled_pos, scaled_pos)
        src, dst = np.where((dist_mat < 35.0) & (dist_mat > 0))
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long).to(device)
        
        if len(src) > 0:
            pos_src = x_in[src]
            pos_dst = x_in[dst]
            rel_pos = pos_dst - pos_src
            edge_attr = torch.cat([rel_pos, torch.norm(rel_pos, dim=1, keepdim=True)], dim=1)
        else:
            edge_attr = torch.empty((0, 3), dtype=torch.float).to(device)

        graph = Data(x=x_in, edge_index=edge_index, edge_attr=edge_attr)
        with torch.no_grad():
            _, offsets, _, _ = gnn_model(graph)
            
        corrected = scaled_pos + offsets.cpu().numpy()
        group_labels = DBSCAN(eps=30, min_samples=1).fit_predict(corrected)
        
        # 1. 提取当前帧的聚类中心
        unique_labels = [l for l in np.unique(group_labels) if l != -1]
        det_c, det_s = [], []
        for l in unique_labels:
            idx = np.where(group_labels == l)[0]
            det_c.append(np.mean(corrected[idx], axis=0))
            pts_raw = scaled_pos[idx]
            wh = np.percentile(pts_raw, 95, axis=0) - np.percentile(pts_raw, 5, axis=0) if len(pts_raw)>1 else np.array([5,5])
            det_s.append(np.maximum(wh, 3.0))
        
        # 2. 更新跟踪器
        pred_c, pred_id, _ = tracker.update(np.array(det_c), np.array(det_s))

        # 3. 【核心修复】建立聚类标签到跟踪 ID 的映射
        label_to_tid = {}
        if len(det_c) > 0 and len(pred_c) > 0:
            dist_map = cdist(det_c, pred_c)
            r_idx, c_idx = linear_sum_assignment(dist_map)
            for ri, ci in zip(r_idx, c_idx):
                if dist_map[ri, ci] < 60: # 关联阈值
                    label_to_tid[unique_labels[ri]] = pred_id[ci]

        # --- 渲染阶段 ---
        
        # 4. 画群体圈 (Group)
        for l in unique_labels:
            if l in label_to_tid:
                tid = label_to_tid[l]
                members = np.where(group_labels == l)[0]
                if len(members) > 1: # 只有两个及以上才算群
                    group_pts = []
                    for midx in members:
                        u, v = world_to_pixel(raw_pos[midx][0], raw_pos[midx][1], H_inv)
                        group_pts.append([u, v])
                    group_pts = np.array(group_pts)
                    
                    # 绘制半透明群体边界 (光晕效果)
                    overlay = frame.copy()
                    center = np.mean(group_pts, axis=0).astype(int)
                    size = (np.max(group_pts[:,0]) - np.min(group_pts[:,0]) + 30, 
                            np.max(group_pts[:,1]) - np.min(group_pts[:,1]) + 30)
                    cv2.ellipse(overlay, tuple(center), (int(size[0]/2), int(size[1]/2)), 0, 0, 360, (230, 230, 230), -1)
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # 5. 画行人点和渐变轨迹
        for i in range(len(raw_pos)):
            l = group_labels[i]
            if l in label_to_tid:
                tid = label_to_tid[l]
                u, v = world_to_pixel(raw_pos[i][0], raw_pos[i][1], H_inv)
                
                if tid not in colors:
                    colors[tid] = (np.random.randint(150,255), np.random.randint(150,255), np.random.randint(150,255))
                
                # 每个行人独立维护一小段足迹
                hist_key = f"{tid}_{i}"
                if hist_key not in history: history[hist_key] = []
                history[hist_key].append((u, v))
                
                # 绘制极细渐变轨迹
                pts = np.array(history[hist_key][-25:], dtype=np.int32)
                for j in range(len(pts)-1):
                    alpha = (j + 1) / len(pts)
                    base_color = colors[tid]
                    blend_color = [int(bc * alpha + 100 * (1 - alpha)) for bc in base_color]
                    cv2.line(frame, tuple(pts[j]), tuple(pts[j+1]), blend_color, 1, lineType=cv2.LINE_AA)
                
                # 绘制极小点和极小文字
                cv2.circle(frame, (u, v), 2, colors[tid], -1, lineType=cv2.LINE_AA)
                cv2.putText(frame, str(tid), (u+4, v-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        writer.append_data(frame)

    reader.close()
    writer.close()
    print(f"✅ 终极版演示视频已生成: {save_path}")

if __name__ == "__main__":
    visualize_on_video()
