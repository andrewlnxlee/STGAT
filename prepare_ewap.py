# prepare_ewap.py
# 将 EWAP (ETH Walking Pedestrians) 数据集转换为 STGAT 模型可用的 npy 格式
# 数据格式对齐 RadarFileDataset 的 get() 方法

import os
import numpy as np
from collections import defaultdict

# ===========================
# 配置
# ===========================
EWAP_ROOT = os.path.join("datasets", "ewap_dataset")
OUTPUT_ROOT = "./data"
FRAMES_PER_EPISODE = 50  # 与仿真数据保持一致
COORD_SCALE = 50.0       # 将 [~-10, ~15]m 坐标缩放到 [~0, ~1000] 的范围
COORD_OFFSET = np.array([500.0, 500.0])  # 中心偏移，使坐标大致在 [0, 1000] 内

SCENES = {
    "seq_eth":   "test_ewap_eth",
    "seq_hotel": "test_ewap_hotel",
}


def parse_obsmat(filepath):
    """
    解析 obsmat.txt
    格式: [frame_number, pedestrian_ID, pos_x, pos_z, pos_y, v_x, v_z, v_y]
    返回: dict[frame_id] -> list of (ped_id, x, y)
    """
    data = np.loadtxt(filepath)
    frames = defaultdict(list)
    for row in data:
        frame_id = int(row[0])
        ped_id = int(row[1])
        x = row[2]   # pos_x
        y = row[4]   # pos_y (跳过 pos_z)
        frames[frame_id].append((ped_id, x, y))
    return frames


def parse_groups(filepath):
    """
    解析 groups.txt
    每行包含同一群组的行人 ID 列表
    返回: dict[ped_id] -> group_id
    """
    ped_to_group = {}
    group_id = 1
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = list(map(int, line.split()))
            if len(ids) < 2:
                continue  # 单人不算群组
            for pid in ids:
                # 可能重复标注，取第一次出现的分组
                if pid not in ped_to_group:
                    ped_to_group[pid] = group_id
            group_id += 1
    
    return ped_to_group


def convert_scene(scene_name, output_name):
    """转换单个场景"""
    scene_dir = os.path.join(EWAP_ROOT, scene_name)
    obsmat_path = os.path.join(scene_dir, "obsmat.txt")
    groups_path = os.path.join(scene_dir, "groups.txt")
    output_dir = os.path.join(OUTPUT_ROOT, output_name)
    
    # 解析数据
    frames_data = parse_obsmat(obsmat_path)
    # 取消读取群体信息，强制单人成群以评估多目标行人跟踪 (MOT)
    ped_to_group = {} # parse_groups(groups_path)
    
    # 为不在群组中的行人分配独立 group_id
    all_ped_ids = set()
    for peds in frames_data.values():
        for pid, _, _ in peds:
            all_ped_ids.add(pid)
    
    max_group_id = max(ped_to_group.values()) if ped_to_group else 0
    next_group_id = max_group_id + 1
    for pid in sorted(all_ped_ids):
        if pid not in ped_to_group:
            ped_to_group[pid] = next_group_id
            next_group_id += 1
    
    print(f"  场景: {scene_name}")
    print(f"  总帧数: {len(frames_data)}")
    print(f"  总行人: {len(all_ped_ids)}")
    print(f"  群组数 (含独行): {next_group_id - 1}")
    print(f"  纯结伴群组: {max_group_id}")
    
    # 按帧号排序
    sorted_frame_ids = sorted(frames_data.keys())
    
    # 将帧数据按 FRAMES_PER_EPISODE 切片成 episode
    all_frame_dicts = []
    for fid in sorted_frame_ids:
        peds = frames_data[fid]
        
        if len(peds) == 0:
            continue
        
        meas_list = []
        label_list = []
        
        for pid, x, y in peds:
            # 坐标缩放 + 偏移
            scaled_x = x * COORD_SCALE + COORD_OFFSET[0]
            scaled_y = y * COORD_SCALE + COORD_OFFSET[1]
            meas_list.append([scaled_x, scaled_y])
            label_list.append(ped_to_group[pid])
        
        meas = np.array(meas_list)
        labels = np.array(label_list)
        
        # 计算群质心 (按 group_id 分组)
        gt_centers_list = []
        unique_groups = np.unique(labels)
        for gid in unique_groups:
            mask = labels == gid
            group_pts = meas[mask]
            cx = np.mean(group_pts[:, 0])
            cy = np.mean(group_pts[:, 1])
            gt_centers_list.append([gid, cx, cy])
        
        gt_centers = np.array(gt_centers_list)
        
        frame_dict = {
            'meas': meas,
            'labels': labels,
            'gt_centers': gt_centers,
        }
        all_frame_dicts.append(frame_dict)
    
    # 切分成 episode
    os.makedirs(output_dir, exist_ok=True)
    
    num_episodes = max(1, len(all_frame_dicts) // FRAMES_PER_EPISODE)
    episode_count = 0
    
    for start in range(0, len(all_frame_dicts) - FRAMES_PER_EPISODE + 1, FRAMES_PER_EPISODE):
        episode = all_frame_dicts[start:start + FRAMES_PER_EPISODE]
        save_path = os.path.join(output_dir, f"sample_{episode_count:05d}.npy")
        np.save(save_path, episode, allow_pickle=True)
        episode_count += 1
    
    # 如果最后剩余帧不足整个 episode 但有足够帧数 (>10)，也保存
    remaining = len(all_frame_dicts) % FRAMES_PER_EPISODE
    if remaining > 10:
        episode = all_frame_dicts[-remaining:]
        save_path = os.path.join(output_dir, f"sample_{episode_count:05d}.npy")
        np.save(save_path, episode, allow_pickle=True)
        episode_count += 1
    
    print(f"  生成 {episode_count} 个 episode -> {output_dir}")
    
    return episode_count


if __name__ == "__main__":
    print("=" * 60)
    print("EWAP -> STGAT 数据预处理")
    print("=" * 60)
    
    total = 0
    for scene, out_name in SCENES.items():
        print(f"\n--- 处理 {scene} ---")
        n = convert_scene(scene, out_name)
        total += n
    
    print(f"\n✅ 预处理完成! 共生成 {total} 个 episode")
    print(f"   坐标缩放: ×{COORD_SCALE}, 偏移: {COORD_OFFSET}")
    print(f"   每 episode: {FRAMES_PER_EPISODE} 帧")
