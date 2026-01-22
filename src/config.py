# src/config.py
# -*- coding: utf-8 -*-

class Config:
    # 基本配置
    TASK_NAME = 'TFBind8-Exact-v0'
    SEED = 42
    DEVICE = 'cuda'

    # GP sampling config (ROOT style)
    GP_NUM_FUNCTIONS = 8        # Sample n_e = 8 GP functions per epoch
    GP_NUM_POINTS = 1024        # Number of pairs per GP function
    GP_NUM_GRADIENT_STEPS = 100 # Gradient steps
    GP_LEARNING_RATE = 0.1         # GP sampling learning rate
    GP_DELTA_LENGTHSCALE = 0.25 # Lengthscale perturbation range
    GP_DELTA_VARIANCE = 0.25    # Variance perturbation range
    GP_INITIAL_LENGTHSCALE = 6.25
    GP_INITIAL_OUTPUTSCALE = 6.25
    GP_NOISE = 0.01
    GP_NUM_FIT_SAMPLES = 15000  # TFBind8 uses partial samples to fit GP
    GP_THRESHOLD_DIFF = 0.001   # Minimum score difference threshold
    GP_TRAJ_STEPS = 200             # Steps to generate trajectory from GP pairs
    GP_TYPE_INITIAL_POINTS = 'highest'  # 'highest', 'lowest', or other

    # Flow Matching training config
    HIDDEN_DIM = 1024
    FM_BATCH_SIZE = 256
    FM_EPOCHS = 150             # Total training E = 150 epochs
    FM_LR = 1e-3
    INFERENCE_STEPS = 50        # Inference steps
    NUM_TEST_SAMPLES = 128
