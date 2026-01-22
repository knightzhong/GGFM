# src/config.py
# -*- coding: utf-8 -*-

class Config:
    TASK_NAME = 'TFBind8-Exact-v0'
    SEED = 42
    DEVICE = 'cuda'

    # NTK Oracle
    NTK_LENGTH_SCALE = 2.0
    NTK_BETA = 0.01
    NYSTROM_SAMPLES = 8192 

    # Trajectory generation (Phase 1)
    TRAJ_STEPS_DESC = 100
    TRAJ_STEPS_ASC = 100
    TRAJ_LR = 0.005
    LAMBDA_FORWARD = 1.0
    LAMBDA_BACKWARD = 0.1
    LAMBDA_UNCERTAINTY = 0.1
    KAPPA = 0.2
    NUM_SEEDS = 1024
    TRAJECTORY_PATH = 'trajectories_tfbind8.npz'

    # GP sampling config (ROOT style)
    GP_NUM_FUNCTIONS = 8        # Sample n_e = 8 GP functions per epoch
    GP_NUM_POINTS = 1024        # Number of pairs per GP function
    GP_NUM_GRADIENT_STEPS = 100 # Gradient steps
    GP_LEARNING_RATE = 0.05     # GP sampling learning rate
    GP_DELTA_LENGTHSCALE = 0.25 # Lengthscale perturbation range
    GP_DELTA_VARIANCE = 0.25    # Variance perturbation range
    GP_INITIAL_LENGTHSCALE = 6.25
    GP_INITIAL_OUTPUTSCALE = 6.25
    GP_NOISE = 0.01
    GP_NUM_FIT_SAMPLES = 15000  # TFBind8 uses partial samples to fit GP
    GP_THRESHOLD_DIFF = 0.001   # Minimum score difference threshold
    GP_TRAJ_STEPS = 50          # Steps to generate trajectory from GP pairs
    GP_TYPE_INITIAL_POINTS = 'highest'  # 'highest', 'lowest', or other

    # Flow Matching training config
    HIDDEN_DIM = 1024
    FM_BATCH_SIZE = 256
    FM_EPOCHS = 100             # Total training E = 100 epochs
    FM_LR = 1e-3
    INFERENCE_STEPS = 50        # Inference steps
    NUM_TEST_SAMPLES = 128
    
    # Classifier-Free Guidance (CFG) - aligned with ROOT
    CFG_PROB = 0.1              # Probability to mask y condition during training (10%)
    CFG_WEIGHT = 0.0            # CFG enhancement weight during inference (0.0 = ??CFG)
    
    # Target score settings (aligned with ROOT)
    ORACLE_Y_MAX = 1.0          # Oracle theoretical maximum for TFBind8
    ORACLE_Y_MIN = 0.0          # Oracle theoretical minimum for TFBind8
    TARGET_ALPHA = 0.8          # Damping factor for target score (ROOT uses 0.8)
