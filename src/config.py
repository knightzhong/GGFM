# src/config.py

class Config:
    # 任务配置
    TASK_NAME = 'TFBind8-Exact-v0' # 或 'AntMorphology-Exact-v0'
    SEED = 42
    DEVICE = 'cuda' # 或 'cpu'

    # NTK Oracle 配置 (Phase 1)
    NTK_LENGTH_SCALE = 2.0
    NTK_BETA = 0.01
    NYSTROM_SAMPLES = 2000  # 显存优化：诱导点数量

    # 轨迹生成配置 (Phase 1)
    TRAJ_STEPS = 10         # M
    TRAJ_LR = 0.05
    LAMBDA_FORWARD = 1.0    # 目标分数权重
    LAMBDA_BACKWARD = 0.1   # 特征一致性权重
    LAMBDA_UNCERTAINTY = 0.1 # 安全约束权重
    KAPPA = 0.5             # 过滤阈值
    NUM_SEEDS = 1024        # 采样多少个点来生成轨迹

    # Flow Matching 网络配置 (Phase 3)
    HIDDEN_DIM = 256
    FM_BATCH_SIZE = 64
    FM_EPOCHS = 100
    FM_LR = 1e-3

    # 推理配置 (Phase 4)
    INFERENCE_STEPS = 10    # 推理时的 ODE 步数