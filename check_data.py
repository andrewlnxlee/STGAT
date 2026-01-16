import numpy as np
import matplotlib.pyplot as plt
import os
import config

def check_one_sample():
    path = os.path.join(config.DATA_ROOT, 'test', 'sample_00001.npy')
    if not os.path.exists(path):
        print("Run generate_data.py first!")
        return

    data = np.load(path, allow_pickle=True)
    print(f"Loaded {len(data)} frames")
    
    # 提取所有点的坐标和颜色
    all_x, all_y, all_c, all_t = [], [], [], []
    
    for t, frame in enumerate(data):
        if len(frame['meas']) == 0: continue
        all_x.extend(frame['meas'][:, 0])
        all_y.extend(frame['meas'][:, 1])
        all_c.extend(frame['labels'])
        all_t.extend([t] * len(frame['meas']))
        
    all_c = np.array(all_c)
    
    plt.figure(figsize=(10, 10))
    unique_ids = np.unique(all_c)
    
    for uid in unique_ids:
        if uid == 0: continue # 跳过杂波
        mask = all_c == uid
        # 用时间 t 来让颜色渐变，或者直接画出轨迹
        plt.scatter(np.array(all_x)[mask], np.array(all_y)[mask], s=5, label=f"Group {int(uid)}")
        
    plt.title("Accumulated Trajectory (Check Split/Merge)")
    plt.legend()
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.savefig("test.jpg")
    plt.show()

if __name__ == "__main__":
    check_one_sample()