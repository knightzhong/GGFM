# src/config.py
# -*- coding: utf-8 -*-
import os
import yaml


class Config:
    # 基本配置
    TASK_NAME = 'TFBind8-Exact-v0'
    SEED = 42
    DEVICE = 'cuda'

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

    # Flow Matching training config
    HIDDEN_DIM = 1024
    DROPOUT = 0.15
    FM_BATCH_SIZE = 256
    FM_EPOCHS = 150
    FM_LR = 1e-3
    INFERENCE_STEPS = 50
    NUM_TEST_SAMPLES = 128

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
    _apply_if_present(gp_cfg.get('traj_steps'), 'GP_TRAJ_STEPS')

    # 兼容不同字段名
    if 'sampling_from_GP_lr' in gp_cfg:
        _apply_if_present(gp_cfg.get('sampling_from_GP_lr'), 'GP_LEARNING_RATE')
    elif 'learning_rate' in gp_cfg:
        _apply_if_present(gp_cfg.get('learning_rate'), 'GP_LEARNING_RATE')


def _apply_model_config(model_cfg):
    fm_cfg = model_cfg.get('FM', {})
    fm_params = fm_cfg.get('params', {})
    fm_opt = fm_cfg.get('optimizer', {})

    _apply_if_present(fm_params.get('hidden_dim'), 'HIDDEN_DIM')
    _apply_if_present(fm_params.get('dropout'), 'DROPOUT')
    _apply_if_present(fm_params.get('ode_steps'), 'INFERENCE_STEPS')
    _apply_if_present(fm_opt.get('lr'), 'FM_LR')


def _apply_training_config(training_cfg):
    _apply_if_present(training_cfg.get('batch_size'), 'FM_BATCH_SIZE')
    _apply_if_present(training_cfg.get('n_epochs'), 'FM_EPOCHS')


def _apply_testing_config(testing_cfg):
    _apply_if_present(testing_cfg.get('num_candidates'), 'NUM_TEST_SAMPLES')


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _read_yaml(config_path)

    _apply_if_present(cfg.get('seed'), 'SEED')
    _apply_if_present(cfg.get('device'), 'DEVICE')

    task_cfg = cfg.get('task', {})
    _apply_if_present(task_cfg.get('name'), 'TASK_NAME')

    gp_cfg = cfg.get('GP', {})
    _apply_gp_config(gp_cfg)

    model_cfg = cfg.get('model', {})
    _apply_model_config(model_cfg)

    training_cfg = cfg.get('training', {})
    _apply_training_config(training_cfg)

    testing_cfg = cfg.get('testing', {})
    _apply_testing_config(testing_cfg)

    return Config
