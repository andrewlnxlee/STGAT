import pandas as pd
import numpy as np
import os
import shutil
import config
from tqdm import tqdm
from scipy.spatial.distance import cdist

# ================= 最终版配置 =================
# 【核心修改】指定使用 SDP 检测器，这是 MOT17 中最可靠的
DETECTOR_NAME = "SDP"  # 你可以换成 "FRCNN" 或 "DPM" 来对比
SEQUENCE_NAME = "MOT17-04" 

# --- 自动生成路径 ---
BASE_PATH = f"MOT17Labels/train/{SEQUENCE_NAME}-{DETECTOR_NAME}"
GT_PATH = os.path.join(f"MOT17Labels/train/{SEQUENCE_NAME}-GT", "gt", "gt.txt") # GT 路径是固定的
DET_PATH = os.path.join(BASE_PATH, "det", "det.txt")

OUTPUT_DIR = os.path.join(config.DATA_ROOT, "test_mot_public")

# 物理参数
W_IMG, H_IMG = 1920, 1080
RADAR_SCALE = 1000.0

# Top-K 策略，保证每帧都有足够的输入
TOP_K_PER_FRAME = 50 
# ===============================================

def convert():
    print(f"--- 正在使用 {DETECTOR_NAME} 检测器 ---")
    if not os.path.exists(GT_PATH) or not os.path.exists(DET_PATH):
        print(f"❌ 找不到文件，请检查路径:\n  GT: {GT_PATH}\n  DET: {DET_PATH}")
        print("请确保你已下载 MOT17 数据并解压到 mot_data 文件夹")
        return

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 1. 读取 GT
    gt_df = pd.read_csv(GT_PATH, header=None)
    gt_data = gt_df.values
    gt_data = gt_data[gt_data[:, 7] == 1] # 只看行人

    # 2. 读取 Det
    det_df = pd.read_csv(DET_PATH, header=None)
    det_data = det_df.fillna(-999).values
    
    max_frame = int(gt_data[:, 0].max())
    episode_data = []
    
    print(f"正在转换 {max_frame} 帧 (策略: 每帧保留 Top {TOP_K_PER_FRAME})...")
    
    for t in tqdm(range(1, max_frame + 1)):
        frame_info = {'meas': [], 'labels': [], 'gt_centers': []}
        
        # GT
        curr_gt = gt_data[gt_data[:, 0] == t]
        gt_list = []
        for row in curr_gt:
            tid, x, y, w, h = int(row[1]), row[2], row[3], row[4], row[5]
            cx = (x + w/2) / W_IMG * RADAR_SCALE
            cy = (y + h/2) / H_IMG * RADAR_SCALE
            if 0 <= cx <= RADAR_SCALE and 0 <= cy <= RADAR_SCALE:
                frame_info['gt_centers'].append([tid, cx, cy])
                gt_list.append({'id': tid, 'pos': np.array([cx, cy])})

        # Det (Top-K)
        curr_det = det_data[det_data[:, 0] == t]
        if len(curr_det) > 0:
            sort_idx = np.argsort(curr_det[:, 6])[::-1]
            num_to_keep = min(len(curr_det), TOP_K_PER_FRAME)
            top_det = curr_det[sort_idx[:num_to_keep]]
            
            for row in top_det:
                x, y, w, h = row[2], row[3], row[4], row[5]
                mx = (x + w/2) / W_IMG * RADAR_SCALE
                my = (y + h/2) / H_IMG * RADAR_SCALE
                if 0 <= mx <= RADAR_SCALE and 0 <= my <= RADAR_SCALE:
                    frame_info['meas'].append([mx, my])
                    
                    matched_id = 0
                    if len(gt_list) > 0:
                        gt_pos_arr = np.array([g['pos'] for g in gt_list])
                        dists = cdist([[mx, my]], gt_pos_arr)[0]
                        min_idx = np.argmin(dists)
                        if dists[min_idx] < 40.0: 
                            matched_id = gt_list[min_idx]['id']
                    frame_info['labels'].append(matched_id)

        # 格式化
        frame_info['meas'] = np.array(frame_info['meas']) if len(frame_info['meas']) > 0 else np.zeros((0,2))
        frame_info['labels'] = np.array(frame_info['labels']) if len(frame_info['labels']) > 0 else np.zeros((0,))
        frame_info['gt_centers'] = np.array(frame_info['gt_centers']) if len(frame_info['gt_centers']) > 0 else np.zeros((0,3))
        
        episode_data.append(frame_info)

    save_path = os.path.join(OUTPUT_DIR, "sample_mot_public.npy")
    np.save(save_path, episode_data, allow_pickle=True)
    print("✅ 转换完成，这次一定有数据了！")

if __name__ == "__main__":
    convert()