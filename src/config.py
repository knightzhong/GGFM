# src/config.py
# -*- coding: utf-8 -*-
import os
import yaml


class Config:
    # 基本配置
    TASK_NAME = 'TFBind8-Exact-v0'
    SEED = 42
    DEVICE = 'cuda'

    # 与 ROOT 对齐：task 归一化开关
    TASK_NORMALIZE_X = True
    TASK_NORMALIZE_Y = True

    # GP sampling config (ROOT style)
    GP_NUM_FUNCTIONS = 8
    GP_NUM_POINTS = 1024
    GP_NUM_GRADIENT_STEPS = 100
    GP_LEARNING_RATE = 0.05
    GP_DELTA_LENGTHSCALE = 0.25
    GP_DELTA_VARIANCE = 0.25
    GP_INITIAL_LENGTHSCALE = 6.25
    GP_INITIAL_OUTPUTSCALE = 6.25
    GP_NOISE = 0.01
    GP_NUM_FIT_SAMPLES = 15000
    GP_THRESHOLD_DIFF = 0.001
    GP_TRAJ_STEPS = 200
    GP_TYPE_INITIAL_POINTS = 'highest'  # 'highest', 'lowest', or other
    GP_USE_FIXED_BEST_X = True  # True=与 ROOT 一致用固定 top 1024 做 GP 采样；False=用混合池(50% top2000+50% 随机)

    # 模型类型: 'FM' (Flow Matching) 或 'BB' (Brownian Bridge)
    MODEL_TYPE = 'FM'

    # Flow Matching training config
    HIDDEN_DIM = 1024
    DROPOUT = 0.15
    FM_BACKBONE = 'vectorfield'  # 'vectorfield' | 'bbmlp'
    FM_BATCH_SIZE = 256
    FM_EPOCHS = 150
    FM_LR = 1e-3
    LAMBDA_GRAD = 0.05
    LAMBDA_SIGMA = 0.01
    SIGMA_MAX = 0.01
    # 与 BB 损失统一：若 >0，训练时对 drift 目标加噪声 mu_target_noisy = (x_end-x_start) + sigma_t*noise，loss 有下界
    FM_DRIFT_NOISE_SCALE = 0.0  # 0=确定性目标；>0 时 sigma_t 与 t(1-t) 相关，与 BB variance 形式一致
    # 目标是否随 t 变（与 BB 一致）：'constant'=mu_target=v；'schedule'=mu_target=m_t*v+sigma_t*noise，推理时 v=mu_pred/m_t
    FM_DRIFT_TARGET = 'schedule'  # 'constant' | 'schedule'
    FM_SCHEDULE_MAX_VAR = 1.0     # schedule 时与 BB 一致的 max_var，variance_t=2*(m_t-m_t^2)*max_var
    INFERENCE_STEPS = 50
    NUM_TEST_SAMPLES = 128
    # 与 ROOT 一致：推理时目标分数 = (oracle_y_max - mean_y)/std_y * alpha
    TEST_ALPHA = 0.8
    # 评估起点: 'highest'=从最高 128 出发(与 ROOT type_sampling=highest 一致), 'low'=从最低 128 出发(与训练分布一致，易有提升)
    EVAL_TYPE_SAMPLING = 'highest'
    EVAL_PERCENTILE_SAMPLING = 0.2  # type_sampling=low 时，从最低的该比例中随机抽 128（与 ROOT 一致）

    # Brownian Bridge config (当 MODEL_TYPE == 'BB' 时生效)
    BB_NUM_TIMESTEPS = 1000
    BB_MT_TYPE = 'linear'
    BB_MAX_VAR = 1.0
    BB_ETA = 0.2
    BB_LOSS_TYPE = 'l1'
    BB_OBJECTIVE = 'grad'
    BB_SKIP_SAMPLE = True
    BB_SAMPLE_TYPE = 'linear'
    BB_SAMPLE_STEP = 200
    BB_CFG_WEIGHT = 0.0  # classifier-free guidance weight
    BB_CLIP_DENOISED = False
    BB_USE_CFG_TEST = False

    # 与 ROOT 对齐：optimizer / scheduler / EMA / accumulate
    OPT_BETA1 = 0.9
    OPT_WEIGHT_DECAY = 0.0
    LR_SCHEDULER_COOLDOWN = 200
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_MIN_LR = 5e-7
    LR_SCHEDULER_PATIENCE = 200
    LR_SCHEDULER_THRESHOLD = 1e-4
    ACCUMULATE_GRAD_BATCHES = 2
    USE_EMA = True
    EMA_DECAY = 0.995
    EMA_START_STEP = 4000
    EMA_UPDATE_INTERVAL = 8
    VALIDATION_INTERVAL = 20
    SAVE_INTERVAL = 20

    # NTK Oracle config (optional)
    NTK_LENGTH_SCALE = 1.0
    NTK_BETA = 1e-4
    NYSTROM_SAMPLES = -1


def _read_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data or {}


def _apply_if_present(value, attr_name):
    if value is not None:
        setattr(Config, attr_name, value)


def _apply_gp_config(gp_cfg):
    _apply_if_present(gp_cfg.get('num_functions'), 'GP_NUM_FUNCTIONS')
    _apply_if_present(gp_cfg.get('num_points'), 'GP_NUM_POINTS')
    _apply_if_present(gp_cfg.get('num_gradient_steps'), 'GP_NUM_GRADIENT_STEPS')
    _apply_if_present(gp_cfg.get('delta_lengthscale'), 'GP_DELTA_LENGTHSCALE')
    _apply_if_present(gp_cfg.get('delta_variance'), 'GP_DELTA_VARIANCE')
    _apply_if_present(gp_cfg.get('initial_lengthscale'), 'GP_INITIAL_LENGTHSCALE')
    _apply_if_present(gp_cfg.get('initial_outputscale'), 'GP_INITIAL_OUTPUTSCALE')
    _apply_if_present(gp_cfg.get('noise'), 'GP_NOISE')
    _apply_if_present(gp_cfg.get('num_fit_samples'), 'GP_NUM_FIT_SAMPLES')
    _apply_if_present(gp_cfg.get('threshold_diff'), 'GP_THRESHOLD_DIFF')
    _apply_if_present(gp_cfg.get('type_of_initial_points'), 'GP_TYPE_INITIAL_POINTS')
    _apply_if_present(gp_cfg.get('use_fixed_best_x_for_gp'), 'GP_USE_FIXED_BEST_X')
    _apply_if_present(gp_cfg.get('traj_steps'), 'GP_TRAJ_STEPS')

    # 兼容不同字段名
    if 'sampling_from_GP_lr' in gp_cfg:
        _apply_if_present(gp_cfg.get('sampling_from_GP_lr'), 'GP_LEARNING_RATE')
    elif 'learning_rate' in gp_cfg:
        _apply_if_present(gp_cfg.get('learning_rate'), 'GP_LEARNING_RATE')


def _apply_model_config(model_cfg):
    # 模型类型切换：FM / FlowMatching -> FM，BB / BBDM -> BB
    model_type = model_cfg.get('model_type')
    if model_type is not None:
        s = str(model_type).upper()
        if s in ('BBDM', 'BROWNIANBRIDGE'):
            setattr(Config, 'MODEL_TYPE', 'BB')
        elif s in ('FM', 'FLOWMATCHING', 'FLOW_MATCHING'):
            setattr(Config, 'MODEL_TYPE', 'FM')
        else:
            setattr(Config, 'MODEL_TYPE', s)

    fm_cfg = model_cfg.get('FM', {})
    fm_params = fm_cfg.get('params', {})
    fm_opt = fm_cfg.get('optimizer', {})

    _apply_if_present(fm_params.get('hidden_dim'), 'HIDDEN_DIM')
    _apply_if_present(fm_params.get('dropout'), 'DROPOUT')
    _apply_if_present(fm_params.get('backbone'), 'FM_BACKBONE')
    _apply_if_present(fm_params.get('lambda_grad'), 'LAMBDA_GRAD')
    _apply_if_present(fm_params.get('lambda_sigma'), 'LAMBDA_SIGMA')
    _apply_if_present(fm_params.get('sigma_max'), 'SIGMA_MAX')
    _apply_if_present(fm_params.get('ode_steps'), 'INFERENCE_STEPS')
    _apply_if_present(fm_opt.get('lr'), 'FM_LR')

    # Brownian Bridge 参数
    bb_cfg = model_cfg.get('BB', {})
    bb_params = bb_cfg.get('params', {})
    bb_opt = bb_cfg.get('optimizer', {})
    bb_sche = bb_cfg.get('lr_scheduler', {})
    if bb_params:
        mlp_p = bb_params.get('MLPParams', {})
        _apply_if_present(mlp_p.get('hidden_size') or bb_params.get('hidden_size'), 'HIDDEN_DIM')
        _apply_if_present(bb_params.get('num_timesteps'), 'BB_NUM_TIMESTEPS')
        _apply_if_present(bb_params.get('mt_type'), 'BB_MT_TYPE')
        _apply_if_present(bb_params.get('max_var'), 'BB_MAX_VAR')
        _apply_if_present(bb_params.get('eta'), 'BB_ETA')
        _apply_if_present(bb_params.get('loss_type'), 'BB_LOSS_TYPE')
        _apply_if_present(bb_params.get('objective'), 'BB_OBJECTIVE')
        _apply_if_present(bb_params.get('skip_sample'), 'BB_SKIP_SAMPLE')
        _apply_if_present(bb_params.get('sample_type'), 'BB_SAMPLE_TYPE')
        _apply_if_present(bb_params.get('sample_step'), 'BB_SAMPLE_STEP')
        _apply_if_present(bb_opt.get('lr'), 'FM_LR')
        _apply_if_present(bb_opt.get('beta1'), 'OPT_BETA1')
        _apply_if_present(bb_opt.get('weight_decay'), 'OPT_WEIGHT_DECAY')
        _apply_if_present(bb_sche.get('cooldown'), 'LR_SCHEDULER_COOLDOWN')
        _apply_if_present(bb_sche.get('factor'), 'LR_SCHEDULER_FACTOR')
        _apply_if_present(bb_sche.get('min_lr'), 'LR_SCHEDULER_MIN_LR')
        _apply_if_present(bb_sche.get('patience'), 'LR_SCHEDULER_PATIENCE')
        _apply_if_present(bb_sche.get('threshold'), 'LR_SCHEDULER_THRESHOLD')

    # ROOT 风格 EMA 配置
    ema_cfg = model_cfg.get('EMA', {})
    if ema_cfg:
        _apply_if_present(ema_cfg.get('use_ema'), 'USE_EMA')
        _apply_if_present(ema_cfg.get('ema_decay'), 'EMA_DECAY')
        _apply_if_present(ema_cfg.get('start_ema_step'), 'EMA_START_STEP')
        _apply_if_present(ema_cfg.get('update_ema_interval'), 'EMA_UPDATE_INTERVAL')


def _apply_training_config(training_cfg):
    _apply_if_present(training_cfg.get('batch_size'), 'FM_BATCH_SIZE')
    _apply_if_present(training_cfg.get('n_epochs'), 'FM_EPOCHS')
    _apply_if_present(training_cfg.get('val_frac'), 'VAL_FRAC')
    _apply_if_present(training_cfg.get('use_classifier_free_guidance'), 'USE_CFG_TRAINING')
    _apply_if_present(training_cfg.get('classifier_free_guidance_prob'), 'CFG_PROB')
    _apply_if_present(training_cfg.get('accumulate_grad_batches'), 'ACCUMULATE_GRAD_BATCHES')
    _apply_if_present(training_cfg.get('validation_interval'), 'VALIDATION_INTERVAL')
    _apply_if_present(training_cfg.get('save_interval'), 'SAVE_INTERVAL')


def _apply_testing_config(testing_cfg):
    _apply_if_present(testing_cfg.get('num_candidates'), 'NUM_TEST_SAMPLES')
    _apply_if_present(testing_cfg.get('classifier_free_guidance_weight'), 'BB_CFG_WEIGHT')
    _apply_if_present(testing_cfg.get('alpha'), 'TEST_ALPHA')
    _apply_if_present(testing_cfg.get('type_sampling'), 'EVAL_TYPE_SAMPLING')
    _apply_if_present(testing_cfg.get('percentile_sampling'), 'EVAL_PERCENTILE_SAMPLING')
    _apply_if_present(testing_cfg.get('clip_denoised'), 'BB_CLIP_DENOISED')
    _apply_if_present(testing_cfg.get('use_classifier_free_guidance'), 'BB_USE_CFG_TEST')


def _apply_task_config(task_cfg):
    _apply_if_present(task_cfg.get('name'), 'TASK_NAME')
    _apply_if_present(task_cfg.get('normalize_x'), 'TASK_NORMALIZE_X')
    _apply_if_present(task_cfg.get('normalize_y'), 'TASK_NORMALIZE_Y')


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _read_yaml(config_path)

    _apply_if_present(cfg.get('seed'), 'SEED')
    _apply_if_present(cfg.get('device'), 'DEVICE')

    task_cfg = cfg.get('task', {})
    _apply_task_config(task_cfg)

    gp_cfg = cfg.get('GP', {})
    _apply_gp_config(gp_cfg)

    model_cfg = cfg.get('model', {})
    _apply_model_config(model_cfg)

    training_cfg = cfg.get('training', {})
    _apply_training_config(training_cfg)

    testing_cfg = cfg.get('testing', {})
    _apply_testing_config(testing_cfg)

    return Config
