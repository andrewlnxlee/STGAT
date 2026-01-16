import numpy as np
import matplotlib.pyplot as plt
import os
import config
NUM=3
def viz_one_sample():
    # 读取刚生成的第 0 个测试样本
    path = os.path.join(config.DATA_ROOT, 'test', f'sample_0000{NUM}.npy')
    if not os.path.exists(path):
        print("请先运行 generate_data.py")
        return

    data = np.load(path, allow_pickle=True)
    
    plt.figure(figsize=(10, 10))
    
    # 颜色表
    colors = {1: 'blue', 2: 'orange', 3: 'green'}
    
    print("正在绘制累积轨迹...")
    # 我们每隔 2 帧画一次，避免太密集
    for t in range(0, len(data), 2):
        frame = data[t]
        if len(frame['meas']) == 0: continue
        
        meas = frame['meas']
        labels = frame['labels']
        
        for uid in np.unique(labels):
            if uid == 0: continue # 不画杂波
            mask = labels == uid
            pts = meas[mask]
            
            # 使用 alpha 透明度模拟时间流逝（越新的越深）
            alpha = 0.1 + 0.9 * (t / len(data))
            
            plt.scatter(pts[:, 0], pts[:, 1], c=colors.get(uid, 'black'), 
                        s=10, alpha=alpha, edgecolors='none')

    plt.title("Accumulated Trajectory: Natural Merge/Split")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.grid(True)
    plt.savefig(f"tesd{NUM}.jpg")
    plt.show()

if __name__ == "__main__":
    viz_one_sample()