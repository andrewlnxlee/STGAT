# config.py

# --- 数据生成配置 ---
DATA_ROOT = "./data"
XISHU=0.1
NUM_TRAIN_SAMPLES = 2000  # 训练集样本数 (每个样本是一个几十帧的片段)
NUM_VAL_SAMPLES = 200     # 验证集
NUM_TEST_SAMPLES = 100   # 测试集
FRAMES_PER_SAMPLE = 50    # 每个样本包含多少帧 (短片段利于训练)
MAX_GROUPS = 5            # 场景中最大群数量

# --- 模型配置 ---
INPUT_DIM = 2     # x, y
EDGE_DIM = 3      # dx, dy, dist
HIDDEN_DIM = 64

# --- 训练配置 ---
BATCH_SIZE = 1    # 图网络建议 Batch=1 (指一次处理一个时序图序列) 或使用 PyG Batch
LEARNING_RATE = 0.001
EPOCHS = 50
MODEL_SAVE_PATH = "best_model_v4.pth"
DEVICE = "cuda"   # 或 "cpu"