import numpy as np
import matplotlib.pyplot as plt
import os
import config

# 读取刚生成的 MOT 数据
DATA_PATH = os.path.join(config.DATA_ROOT, "test_mot_public", "sample_mot_public.npy")

def check():
    if not os.path.exists(DATA_PATH):
        print("找不到数据，请先运行转换脚本")
        return

    data = np.load(DATA_PATH, allow_pickle=True)
    print(f"数据加载成功，共 {len(data)} 帧")

    # 随机找一帧有数据的
    target_frame = None
    for i, frame in enumerate(data):
        if len(frame['meas']) > 0 and len(frame['gt_centers']) > 0:
            target_frame = frame
            print(f"正在检查第 {i} 帧...")
            break
    
    if target_frame is None:
        print("❌ 灾难！所有帧都没有匹配的数据（要么没GT，要么没Meas）")
        return

    # 绘图
    gt = target_frame['gt_centers']
    meas = target_frame['meas']

    plt.figure(figsize=(10, 10))
    
    # 画 GT
    plt.scatter(gt[:, 1], gt[:, 2], c='green', marker='*', s=200, label='Ground Truth')
    
    # 画 Meas
    plt.scatter(meas[:, 0], meas[:, 1], c='gray', alpha=0.6, s=50, label='Public Detection')

    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.legend()
    plt.title("Alignment Check: GT vs Detection")
    plt.savefig("mot_debug.png")
    print("已保存诊断图: mot_debug.png")

if __name__ == "__main__":
    check()