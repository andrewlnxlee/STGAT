import pandas as pd
import numpy as np
import os
import shutil
import config
from tqdm import tqdm
from scipy.spatial.distance import cdist

# ================= 强力修复配置 =================
# 建议改用 SDP (如果有)，如果只有 FRCNN 就用 FRCNN
# 路径请根据你服务器实际情况修改
BASE_PATH = "MOT17Labels/train/MOT17-04-FRCNN"
GT_PATH = os.path.join(BASE_PATH, "gt", "gt.txt")
DET_PATH = os.path.join(BASE_PATH, "det", "det.txt")

OUTPUT_DIR = os.path.join(config.DATA_ROOT, "test_mot_public")

W_IMG, H_IMG = 1920, 1080
RADAR_SCALE = 1000.0

# 【核心修改】不看阈值，只看排名
# 每一帧强制保留 Score 最高的 K 个检测结果
TOP_K_PER_FRAME = 50 
# ===============================================

def convert():
    if not os.path.exists(GT_PATH) or not os.path.exists(DET_PATH):
        print(f"❌ 找不到文件，请检查路径:\n{GT_PATH}\n{DET_PATH}")
        return

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("--- 正在执行 Top-K 强制转换 ---")
    
    # 1. 读取 GT
    gt_df = pd.read_csv(GT_PATH, header=None)
    gt_data = gt_df.values
    # 过滤掉非人类 (class!=1)
    gt_data = gt_data[gt_data[:, 7] == 1]
    
    max_frame = int(gt_data[:, 0].max())
    
    # 2. 读取 Det
    print("读取检测结果...")
    det_df = pd.read_csv(DET_PATH, header=None)
    # 填补可能的 NaN
    det_df = det_df.fillna(-999) 
    det_data = det_df.values
    
    # 打印原始数量
    print(f"原始检测框总数: {len(det_data)}")
    
    episode_data = []
    total_dets_kept = 0
    
    print(f"正在转换 {max_frame} 帧 (策略: 每帧保留 Top {TOP_K_PER_FRAME})...")
    
    for t in tqdm(range(1, max_frame + 1)):
        frame_info = {'meas': [], 'labels': [], 'gt_centers': []}
        
        # --- A. 处理 GT ---
        curr_gt = gt_data[gt_data[:, 0] == t]
        gt_list = []
        for row in curr_gt:
            tid = int(row[1])
            x, y, w, h = row[2], row[3], row[4], row[5]
            cx = (x + w/2) / W_IMG * RADAR_SCALE
            cy = (y + h/2) / H_IMG * RADAR_SCALE
            
            if 0 <= cx <= RADAR_SCALE and 0 <= cy <= RADAR_SCALE:
                frame_info['gt_centers'].append([tid, cx, cy])
                gt_list.append({'id': tid, 'pos': np.array([cx, cy])})

        # --- B. 处理 Det (Top-K 逻辑) ---
        # 1. 取出当前帧所有检测
        curr_det = det_data[det_data[:, 0] == t]
        
        if len(curr_det) > 0:
            # 2. 按置信度 (第6列) 从大到小排序
            # argsort 返回从小到大的索引，[::-1] 反转
            sort_idx = np.argsort(curr_det[:, 6])[::-1]
            
            # 3. 取前 K 个 (如果不足 K 个就全取)
            num_to_keep = min(len(curr_det), TOP_K_PER_FRAME)
            top_det = curr_det[sort_idx[:num_to_keep]]
            
            for row in top_det:
                x, y, w, h = row[2], row[3], row[4], row[5]
                mx = (x + w/2) / W_IMG * RADAR_SCALE
                my = (y + h/2) / H_IMG * RADAR_SCALE
                
                if 0 <= mx <= RADAR_SCALE and 0 <= my <= RADAR_SCALE:
                    frame_info['meas'].append([mx, my])
                    total_dets_kept += 1
                    
                    # 匹配 Label (Purity用)
                    matched_id = 0
                    if len(gt_list) > 0:
                        gt_pos_arr = np.array([g['pos'] for g in gt_list])
                        dists = cdist([[mx, my]], gt_pos_arr)[0]
                        min_idx = np.argmin(dists)
                        # 距离阈值放宽到 40
                        if dists[min_idx] < 40.0: 
                            matched_id = gt_list[min_idx]['id']
                    frame_info['labels'].append(matched_id)

        # 格式转换
        frame_info['meas'] = np.array(frame_info['meas']) if len(frame_info['meas']) > 0 else np.zeros((0,2))
        frame_info['labels'] = np.array(frame_info['labels']) if len(frame_info['labels']) > 0 else np.zeros((0,))
        frame_info['gt_centers'] = np.array(frame_info['gt_centers']) if len(frame_info['gt_centers']) > 0 else np.zeros((0,3))
        
        episode_data.append(frame_info)

    save_path = os.path.join(OUTPUT_DIR, "sample_mot_public.npy")
    np.save(save_path, episode_data, allow_pickle=True)
    print(f"✅ 转换完成。")
    print(f"共保留有效检测点: {total_dets_kept} (平均每帧 {total_dets_kept/max_frame:.1f} 个)")

if __name__ == "__main__":
    convert()