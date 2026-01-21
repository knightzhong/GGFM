# src/config.py

class Config:
    TASK_NAME = 'TFBind8-Exact-v0'
    SEED = 42
    DEVICE = 'cuda'

    # NTK Oracle 保持不变
    NTK_LENGTH_SCALE = 2.0
    NTK_BETA = 0.01
    NYSTROM_SAMPLES = 8192 

    # 轨迹生成重构 (Phase 1)
    TRAJ_STEPS_DESC = 500    # 向下寻找低分起点的步数
    TRAJ_STEPS_ASC = 800     # 向上寻找高分终点的步数
    TRAJ_LR = 0.005          # 大幅降低步长，保证 1300 步的平滑性
    LAMBDA_FORWARD = 1.0    # 目标分数权重
    LAMBDA_BACKWARD = 0.1   # 特征一致性权重
    LAMBDA_UNCERTAINTY = 0.1 # 安全约束权重
    KAPPA = 0.2             # 过滤阈值
    NUM_SEEDS = 1024        # 采样多少个点来生成轨迹
    TRAJECTORY_PATH = 'trajectories_tfbind8.npz'

    # Flow Matching 训练配置
    HIDDEN_DIM = 1024
    FM_BATCH_SIZE = 256
    FM_EPOCHS = 200          # 离线数据更丰富，Epoch 可以适当减少
    FM_LR = 1e-3
    INFERENCE_STEPS = 20     # 推理步数稍增，提高精度
    NUM_TEST_SAMPLES = 128