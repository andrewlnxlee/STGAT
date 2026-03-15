import os
import numpy as np
import math
import random
import config
from tqdm import tqdm

class ActiveInteractionScenarioEngine:
    def __init__(self, num_frames=50):
        self.num_frames = num_frames
        self.area_size = (1000, 1000)
        self.clutter_rate = 8
        self.detection_prob = 0.95
        
        # --- 修改点1：大幅提高速度上限，确保能飞完半张地图 ---
        self.min_speed = 10.0   # 原来是 6.0
        self.max_speed = 25.0   # 原来是 15.0

    def generate_episode(self):
        """
        场景概率调整：稍微增加混合场景的概率，让画面更丰富
        """
        prob = random.random()
        if prob < 0.4:
            return self._run_converge_scenario() # 40% 纯汇聚
        elif prob < 0.7:
            return self._run_diverge_scenario()  # 30% 纯分裂
        else:
            return self._run_mixed_scenario()    # 30% 混合

    # ==========================
    # 辅助：智能生成群 (保证能飞到终点)
    # ==========================
    def _spawn_group_aiming_at(self, group_id, target_pos, start_area=None):
        """
        生成一个群，初始速度朝向 target_pos。
        关键逻辑：检查距离，如果太远飞不到，就拉近一点。
        """
        # 1. 确定初始随机位置
        if start_area:
            x = random.uniform(start_area[0], start_area[1])
            y = random.uniform(start_area[2], start_area[3])
            pos = np.array([x, y])
        else:
            # 边缘随机生成
            side = random.randint(0, 3)
            if side == 0:   pos = np.array([-50.0, random.uniform(0, 1000)])
            elif side == 1: pos = np.array([1050.0, random.uniform(0, 1000)])
            elif side == 2: pos = np.array([random.uniform(0, 1000), -50.0])
            else:           pos = np.array([random.uniform(0, 1000), 1050.0])

        # 2. --- 修改点2：距离校验与修正 ---
        # 计算到目标的距离
        dist_to_target = np.linalg.norm(target_pos - pos)
        
        # 估算理论最大航程 (留 10 帧的余量用于合并动作)
        avg_speed = (self.min_speed + self.max_speed) / 2
        max_flyable_dist = avg_speed * (self.num_frames - 10)
        
        # 如果距离太远，强行拉近出生点
        if dist_to_target > max_flyable_dist:
            # 在连线上插值，找一个新的起点
            ratio = max_flyable_dist / (dist_to_target + 1e-5)
            # 新起点 = 目标点 + (旧起点-目标点) * 比例
            # 即从目标点往回倒推 max_flyable_dist 的距离
            pos = target_pos + (pos - target_pos) * ratio
            
            # 稍微加一点随机扰动，别都在直线上
            pos += np.random.randn(2) * 20.0

        # 3. 计算速度
        dir_vec = target_pos - pos
        dir_vec /= (np.linalg.norm(dir_vec) + 1e-5)
        
        # 增加随机偏角
        angle_noise = random.uniform(-0.15, 0.15)
        c, s = math.cos(angle_noise), math.sin(angle_noise)
        dir_vec = np.array([dir_vec[0]*c - dir_vec[1]*s, dir_vec[0]*s + dir_vec[1]*c])
        
        # 既然是为了合并，尽量给个高速度
        speed = random.uniform(self.min_speed + 5, self.max_speed) 
        vel = dir_vec * speed
        
        num_members = random.randint(5, 15)
        
        return {
            'c': pos,
            'v': vel,
            'offsets': np.random.randn(num_members, 2) * 8.0,
            'member_vels': np.zeros((num_members, 2)),
            'active': True,
            'target': target_pos, 
            'role': 'merger'
        }

    # ==========================
    # 场景 1: 多向汇聚 (Converge / Merge)
    # ==========================
    def _run_converge_scenario(self):
        groups = {}
        # 随机设置 1 到 2 个集结点
        num_rps = random.choice([1, 1, 2]) 
        rps = []
        for _ in range(num_rps):
            # 集结点尽量在中心区域，方便四周汇聚
            rps.append(np.array([random.uniform(300, 700), random.uniform(300, 700)]))
            
        num_groups = random.randint(2, 5)
        for i in range(num_groups):
            assigned_rp = random.choice(rps)
            groups[i+1] = self._spawn_group_aiming_at(i+1, assigned_rp)
            
        merged_pairs = set()
        
        episode_data = []
        for t in range(self.num_frames):
            frame_info = {'meas': [], 'labels': [], 'gt_centers': []}
            
            # --- 交互逻辑 ---
            active_ids = list(groups.keys())
            for i in range(len(active_ids)):
                id1 = active_ids[i]
                if id1 not in groups: continue 
                
                for j in range(i+1, len(active_ids)):
                    id2 = active_ids[j]
                    if id2 not in groups: continue
                    
                    # 距离判定合并 (阈值稍微调大一点，更容易吸附)
                    dist = np.linalg.norm(groups[id1]['c'] - groups[id2]['c'])
                    if dist < 30.0:
                        # 合并
                        g1, g2 = groups[id1], groups[id2]
                        g1['offsets'] = np.vstack([g1['offsets'], g2['offsets']])
                        g1['member_vels'] = np.vstack([g1['member_vels'], g2['member_vels']])
                        # 速度融合
                        g1['v'] = (g1['v'] + g2['v']) * 0.5
                        del groups[id2]
            
            # --- 运动更新 ---
            for gid, g in groups.items():
                self._apply_guidance(g, g['target'])
                self._apply_wander(g)
            
            self._update_members_and_record(groups, frame_info)
            episode_data.append(frame_info)
            
        return episode_data

    # ==========================
    # 场景 2: 穿越分裂 (Diverge / Split)
    # ==========================
    def _run_diverge_scenario(self):
        groups = {}
        num_groups = random.randint(1, 3)
        
        for i in range(num_groups):
            # 随机起点终点
            if random.random() < 0.5: 
                y_start = random.uniform(100, 900)
                start_pos = np.array([-50.0, y_start])
                target_pos = np.array([1050.0, random.uniform(100, 900)])
            else: 
                x_start = random.uniform(100, 900)
                start_pos = np.array([x_start, -50.0])
                target_pos = np.array([random.uniform(100, 900), 1050.0])
                
            g = self._spawn_group_aiming_at(i+1, target_pos)
            g['c'] = start_pos 
            # 强制修正：如果自动计算的出生点还是太远，就不管逻辑了，强制拉近
            # (虽然_spawn里已经处理了，但这属于穿越场景，起点固定，所以再校验一次)
            dist_total = np.linalg.norm(target_pos - start_pos)
            if dist_total > 1200: # 太远了
                g['v'] *= 1.5 # 既然路远，那就飞快点！
                
            if len(g['offsets']) < 12:
                extra = np.random.randn(8, 2) * 8.0
                g['offsets'] = np.vstack([g['offsets'], extra])
                g['member_vels'] = np.vstack([g['member_vels'], np.zeros_like(extra)])
                
            g['split_time'] = random.randint(int(self.num_frames*0.3), int(self.num_frames*0.6))
            groups[i+1] = g
            
        episode_data = []
        next_id = max(groups.keys()) + 1
        
        for t in range(self.num_frames):
            frame_info = {'meas': [], 'labels': [], 'gt_centers': []}
            
            # --- 分裂检测 ---
            current_ids = list(groups.keys())
            for gid in current_ids:
                g = groups[gid]
                if 'split_time' in g and t == g['split_time']:
                    parent = g
                    half = len(parent['offsets']) // 2
                    if half < 3: continue 
                    
                    vel_norm = parent['v'] / (np.linalg.norm(parent['v']) + 1e-5)
                    orth_vec = np.array([-vel_norm[1], vel_norm[0]]) 
                    if random.random() > 0.5: orth_vec *= -1
                    
                    child = {
                        'c': parent['c'].copy(),
                        'v': parent['v'].copy(),
                        'offsets': parent['offsets'][half:].copy(),
                        'member_vels': parent['member_vels'][half:].copy(),
                        'active': True,
                        # 分裂后给个新目标，稍微偏离原目标
                        'target': parent['target'] + orth_vec * 400 
                    }
                    child['v'] += orth_vec * 4.0 # 侧向推力加大
                    
                    parent['offsets'] = parent['offsets'][:half]
                    parent['member_vels'] = parent['member_vels'][:half]
                    parent['v'] -= orth_vec * 2.0 
                    
                    groups[next_id] = child
                    next_id += 1
                    del parent['split_time']
            
            for gid, g in groups.items():
                self._apply_guidance(g, g['target'])
                self._apply_wander(g)
            
            self._update_members_and_record(groups, frame_info)
            episode_data.append(frame_info)
            
        return episode_data

    # ==========================
    # 场景 3: 混合大乱斗 (Mixed) - 速度加快版
    # ==========================
    def _run_mixed_scenario(self):
        groups = {}
        # 1. 必有: 分裂群
        start_pos = np.array([100.0, 200.0])
        target_pos = np.array([900.0, 800.0])
        g1 = self._spawn_group_aiming_at(1, target_pos)
        g1['c'] = start_pos
        g1['split_time'] = 20
        # 确保速度够快
        g1['v'] = g1['v'] / np.linalg.norm(g1['v']) * 20.0 
        # 加人
        extra = np.random.randn(12, 2) * 8.0
        g1['offsets'] = np.vstack([g1['offsets'], extra])
        g1['member_vels'] = np.vstack([g1['member_vels'], np.zeros_like(extra)])
        groups[1] = g1
        
        # 2. 必有: 汇聚群 (相向而行，速度极快)
        center = np.array([500.0, 500.0])
        
        g2 = self._spawn_group_aiming_at(2, center)
        g2['c'] = np.array([200.0, 800.0]) 
        g2['v'] = (center - g2['c']); g2['v'] /= np.linalg.norm(g2['v']); g2['v'] *= 22.0 # 高速
        groups[2] = g2
        
        g3 = self._spawn_group_aiming_at(3, center)
        g3['c'] = np.array([800.0, 200.0])
        g3['v'] = (center - g3['c']); g3['v'] /= np.linalg.norm(g3['v']); g3['v'] *= 22.0 # 高速
        groups[3] = g3
        
        next_id = 4
        episode_data = []
        
        for t in range(self.num_frames):
            frame_info = {'meas': [], 'labels': [], 'gt_centers': []}
            
            # 分裂
            if 1 in groups and t == groups[1].get('split_time', -1):
                parent = groups[1]
                half = len(parent['offsets']) // 2
                vel_norm = parent['v'] / (np.linalg.norm(parent['v']) + 1e-5)
                orth_vec = np.array([-vel_norm[1], vel_norm[0]])
                
                child = {
                    'c': parent['c'].copy(),
                    'v': parent['v'].copy() + orth_vec * 5.0, # 强力侧推
                    'offsets': parent['offsets'][half:].copy(),
                    'member_vels': parent['member_vels'][half:].copy(),
                    'active': True,
                    'target': parent['target']
                }
                parent['offsets'] = parent['offsets'][:half]
                parent['member_vels'] = parent['member_vels'][:half]
                groups[next_id] = child
                next_id += 1
            
            # 合并 (ID 2 和 3)
            if 2 in groups and 3 in groups:
                 dist = np.linalg.norm(groups[2]['c'] - groups[3]['c'])
                 if dist < 40.0: # 加大合并阈值
                     g_keep = groups[2]
                     g_del = groups[3]
                     g_keep['offsets'] = np.vstack([g_keep['offsets'], g_del['offsets']])
                     g_keep['member_vels'] = np.vstack([g_keep['member_vels'], g_del['member_vels']])
                     del groups[3]

            for gid, g in groups.items():
                self._apply_guidance(g, g['target'])
                self._apply_wander(g)

            self._update_members_and_record(groups, frame_info)
            episode_data.append(frame_info)
            
        return episode_data

    # ==========================
    # 核心物理引擎 (针对合并优化)
    # ==========================
    def _apply_guidance(self, group, target_pos):
        """PD 导引：平滑转向目标，但允许加力追赶"""
        desired_vec = target_pos - group['c']
        dist = np.linalg.norm(desired_vec)
        if dist > 1.0: desired_vec /= dist
        
        curr_speed = np.linalg.norm(group['v'])
        
        # --- 修改点3：距离远时自动加速 (加力模式) ---
        target_speed = curr_speed
        if dist > 300: # 如果还很远
             target_speed = max(curr_speed, self.max_speed * 1.2) # 允许超速 20%
        elif dist < 100:
             target_speed = min(curr_speed, self.max_speed * 0.8) # 接近时减速，方便捕获
             
        desired_vel = desired_vec * target_speed
        
        # --- 修改点4：减小转弯惯性，让它转向更灵敏 ---
        inertia = 0.85 # 原来是 0.95 (太笨重)，现在改小，转向更快
        
        group['v'] = group['v'] * inertia + desired_vel * (1 - inertia)
        
        # 速度钳制
        actual_speed = np.linalg.norm(group['v'])
        if actual_speed > self.max_speed * 1.5:
             group['v'] = group['v'] / actual_speed * (self.max_speed * 1.5)
        
    def _apply_wander(self, group):
        """叠加随机扰动"""
        speed = np.linalg.norm(group['v'])
        speed += random.uniform(-1.0, 1.0) # 加减速更剧烈一点
        speed = np.clip(speed, self.min_speed, self.max_speed * 1.5)
        
        angle = random.uniform(-0.03, 0.03) # 偏航稍微减小，避免破坏导引
        c, s = math.cos(angle), math.sin(angle)
        vx, vy = group['v']
        group['v'] = np.array([vx*c - vy*s, vx*s + vy*c]) / (np.linalg.norm([vx,vy])+1e-5) * speed
        
        group['c'] += group['v']

    def _update_members_and_record(self, groups, frame_info):
        # 杂波
        n_clutter = np.random.poisson(self.clutter_rate)
        for _ in range(n_clutter):
            frame_info['meas'].append([random.uniform(0, 1000), random.uniform(0, 1000)])
            frame_info['labels'].append(0)

        for gid, g in groups.items():
            if 'member_vels' not in g: g['member_vels'] = np.zeros_like(g['offsets'])
            
            force = -0.05 * g['offsets']
            g['member_vels'] += force + np.random.randn(*g['offsets'].shape) * 0.5
            g['member_vels'] *= 0.9
            g['offsets'] += g['member_vels']
            
            frame_info['gt_centers'].append([gid, g['c'][0], g['c'][1]])
            
            true_pts = g['c'] + g['offsets']
            for pt in true_pts:
                # 视野检查
                if pt[0] < 0 or pt[0] > 1000 or pt[1] < 0 or pt[1] > 1000:
                    continue 
                
                if random.random() < self.detection_prob:
                    noise = np.random.randn(2) * 1.5
                    frame_info['meas'].append(pt + noise)
                    frame_info['labels'].append(gid)
        
        if len(frame_info['meas']) > 0:
            frame_info['meas'] = np.array(frame_info['meas'])
            frame_info['labels'] = np.array(frame_info['labels'])
            frame_info['gt_centers'] = np.array(frame_info['gt_centers'])
        else:
            frame_info['meas'] = np.zeros((0,2))
            frame_info['labels'] = np.zeros((0,))
            frame_info['gt_centers'] = np.zeros((0,3))

def save_dataset(split_name, num_samples):
    folder = os.path.join(config.DATA_ROOT, split_name)
    if os.path.exists(folder):
        import shutil
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    
    sim = ActiveInteractionScenarioEngine(num_frames=config.FRAMES_PER_SAMPLE)
    print(f"Generating {split_name} data ({num_samples} HIGH SPEED INTERACTION scenarios)...")
    for i in tqdm(range(num_samples)):
        episode = sim.generate_episode()
        save_path = os.path.join(folder, f"sample_{i:05d}.npy")
        np.save(save_path, episode, allow_pickle=True)

if __name__ == "__main__":
    save_dataset("train", config.NUM_TRAIN_SAMPLES)
    save_dataset("val", config.NUM_VAL_SAMPLES)
    save_dataset("test", config.NUM_TEST_SAMPLES)