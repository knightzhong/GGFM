# main.py
import argparse
import design_bench
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm.autonotebook import tqdm

from src.config import Config, load_config
from src.utils import set_seed, Normalizer
from src.oracle import NTKOracle
from src.generator import GP, sampling_data_from_GP, create_train_dataloader, create_val_dataloader
# ROOT 没有的功能暂时注释
# from src.generator import generate_trajectories_from_GP_samples
from src.models import VectorFieldNet
from src.brownian_bridge import BrownianBridgeModel
from src.flow import train_cfm_step, train_bb_step, inference_ode, inference_bb
import time
import os


def weights_init(m):
    """与 ROOT runners/utils.py 的 weights_init 完全一致（N(0,0.02) 初始化）"""
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Parameter") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class EMA:
    """与 ROOT_new/runners/base/EMA.py 完全一致"""
    def __init__(self, ema_decay: float):
        super().__init__()
        self.ema_decay = ema_decay
        self.backup = {}
        self.shadow = {}

    def register(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, current_model: nn.Module, with_decay: bool = True):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if with_decay:
                    new_average = (1.0 - self.ema_decay) * param.data + self.ema_decay * self.shadow[name]
                else:
                    new_average = param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def parse_args():
    parser = argparse.ArgumentParser(description="GGFM training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/TfBind8_FlowMatching.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cuda, cpu)",
    )
    return parser.parse_args()


def resolve_config_path(config_path):
    if os.path.isabs(config_path) or os.path.exists(config_path):
        return config_path
    return os.path.join(os.path.dirname(__file__), config_path)

def get_design_bench_data(task_name):
    """
    加载并标准化 Design-Bench 数据，支持离散任务转换
    完全对齐 ROOT 的处理方式
    """
    print(f"Loading task: {task_name}...")
    if task_name != 'TFBind10-Exact-v0':
        task = design_bench.make(task_name)
    else:
        # 显存优化（与 ROOT 一致）
        task = design_bench.make(task_name, dataset_kwargs={"max_samples": 10000})
    
    offline_x = task.x
    logits_shape = None  # 保存 logits 形状信息
    
    if task.is_discrete:
        # 与 ROOT 完全一致：不修改 task 内部状态，使用 to_logits() 得到 logits 表示
        offline_x = task.to_logits(offline_x)
        logits_shape = offline_x.shape  # (N, L, V-1)
        offline_x = offline_x.reshape(offline_x.shape[0], -1)  # (N, L*(V-1))
        print(f"[数据编码] 离散任务：to_logits {logits_shape} -> 展平 {offline_x.shape}")
    else:
        print("[数据编码] 连续任务：直接使用原始数据")
    
    # 计算统计量（与 ROOT 完全一致）
    mean_x = np.mean(offline_x, axis=0)
    std_x = np.std(offline_x, axis=0)
    std_x = np.where(std_x == 0, 1.0, std_x)  # ROOT 使用 == 0，不是 < 1e-6
    offline_x_norm = (offline_x - mean_x) / std_x if getattr(Config, "TASK_NORMALIZE_X", True) else offline_x
    
    # 处理 Y（与 ROOT 一致）
    offline_y = task.y.reshape(-1)  # ROOT 使用 reshape(-1)，不是 reshape(-1, 1)
    mean_y = np.mean(offline_y, axis=0)
    std_y = np.std(offline_y, axis=0)
    
    # 洗牌数据（与 ROOT 一致）
    shuffle_idx = np.random.permutation(offline_x.shape[0])
    offline_x_norm = offline_x_norm[shuffle_idx]
    offline_y = offline_y[shuffle_idx]
    
    # 标准化 Y
    offline_y_norm = (offline_y - mean_y) / std_y if getattr(Config, "TASK_NORMALIZE_Y", True) else offline_y
    
    return task, offline_x_norm, offline_y_norm, mean_x, std_x, mean_y, std_y, logits_shape

# main.py 预处理逻辑
# def preprocess_trajectories(oracle, X_train_norm):
#     device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
#     if os.path.exists(Config.TRAJECTORY_PATH):
#         print(f"Loading cached trajectories from {Config.TRAJECTORY_PATH}")
#         return np.load(Config.TRAJECTORY_PATH)['trajs']

#     print("=== Generating ALL long trajectories (GD reverse + GA) ===")
#     all_indices = np.arange(len(X_train_norm))
#     # 按照你的要求：用全量数据，或者至少 10000 条
#     sample_size = len(all_indices)
#     selected_idx = all_indices
#     # selected_idx = np.random.choice(all_indices, sample_size, replace=False)
    
#     all_valid = []
#     batch_size = 256
#     for i in range(0, sample_size, batch_size):
#         batch_x = X_train_norm[selected_idx[i : i + batch_size]]
#         trajs = generate_long_trajectories(oracle, batch_x, device)
#         all_valid.append(trajs)
#         print(f"Progress: {i + batch_size}/{sample_size}")
        
#     pool = np.concatenate(all_valid, axis=0)
#     np.savez_compressed(Config.TRAJECTORY_PATH, trajs=pool)
#     return pool

def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)
    load_config(config_path)
    if args.seed is not None:
        Config.SEED = args.seed
    if args.device is not None:
        Config.DEVICE = args.device

    # 0. 初始化环境
    print(f"=== GGFM with ROOT GP Sampling: {Config.TASK_NAME} ===")
    print(f"[Config] Using: {config_path}")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # 1. 加载并编码数据（完全对齐 ROOT 的处理方式）
    task, X_train_norm, y_train_norm, mean_x, std_x, mean_y, std_y, logits_shape = get_design_bench_data(Config.TASK_NAME)
    
    # 同步 Normalizer 状态
    x_normalizer = Normalizer(np.zeros((1, X_train_norm.shape[1])))
    x_normalizer.mean, x_normalizer.std, x_normalizer.device = mean_x, std_x, device
    
    # 2. 转换为 Tensor 供 GP 使用（与 ROOT 一致）
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    # y_train_norm 现在是 (N,) 形状，需要转换为 (N, 1) 供 Oracle 使用
    y_train_tensor = torch.FloatTensor(y_train_norm).reshape(-1, 1).to(device)
    
    # 保存原始统计量（用于反标准化）
    mean_x_torch = torch.FloatTensor(mean_x).to(device)
    std_x_torch = torch.FloatTensor(std_x).to(device)
    mean_y_torch = torch.FloatTensor([mean_y]).to(device)
    std_y_torch = torch.FloatTensor([std_y]).to(device)
    
    # 3. 初始化 GP 超参数
    lengthscale = torch.tensor(Config.GP_INITIAL_LENGTHSCALE, device=device)
    variance = torch.tensor(Config.GP_INITIAL_OUTPUTSCALE, device=device)
    noise = torch.tensor(Config.GP_NOISE, device=device)
    mean_prior = torch.tensor(0.0, device=device)
    
    # 4. 选择用于 GP 拟合的初始点（完全对齐 ROOT）
    if Config.GP_TYPE_INITIAL_POINTS == 'highest':
        # ROOT: 固定 1024 个样本，每次全选但顺序不同
        best_indices = torch.argsort(y_train_tensor.view(-1))[-1024:]
        best_x = X_train_tensor[best_indices]
        print(f"[GP Init] Using top 1024 samples for GP sampling (ROOT style: same samples, different order each epoch)")
    elif Config.GP_TYPE_INITIAL_POINTS == 'lowest':
        best_indices = torch.argsort(y_train_tensor.view(-1))[:1024]
        best_x = X_train_tensor[best_indices]
        print(f"[GP Init] Using bottom 1024 samples for GP sampling")
    else:
        best_x = X_train_tensor
        print(f"[GP Init] Using all samples for GP sampling")
    # 1. 预先准备好全量索引 (在循环外)
    all_indices = torch.arange(X_train_tensor.shape[0], device=device)
    # 预先计算好高分索引 (Top 2000 作为一个池子，防止只盯着 Top 1024 过拟合)
    top_k_indices = torch.argsort(y_train_tensor.view(-1), descending=True)[:2000]
    
    # 5. 根据 config 初始化模型 (FM 或 Brownian Bridge)
    input_dim = X_train_norm.shape[1]
    use_bb = getattr(Config, 'MODEL_TYPE', 'FM').upper() == 'BB'
    if use_bb:
        net = BrownianBridgeModel(
            image_size=input_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_timesteps=Config.BB_NUM_TIMESTEPS,
            mt_type=Config.BB_MT_TYPE,
            max_var=Config.BB_MAX_VAR,
            eta=Config.BB_ETA,
            loss_type=Config.BB_LOSS_TYPE,
            objective=Config.BB_OBJECTIVE,
            skip_sample=Config.BB_SKIP_SAMPLE,
            sample_type=Config.BB_SAMPLE_TYPE,
            sample_step=Config.BB_SAMPLE_STEP,
        ).to(device)
        print(f"[Model] Brownian Bridge (BBDM), steps={Config.BB_SAMPLE_STEP}, objective={Config.BB_OBJECTIVE}")
    else:
        net = VectorFieldNet(
            input_dim,
            hidden_dim=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
        ).to(device)
        print("[Model] Flow Matching (FM)")
    # 与 ROOT 对齐：权重初始化
    net.apply(weights_init)
    # 与 ROOT 对齐：优化器配置（Adam beta1=0.9, weight_decay=0）
    optimizer = optim.Adam(
        net.parameters(),
        lr=Config.FM_LR,
        betas=(getattr(Config, "OPT_BETA1", 0.9), 0.999),
        weight_decay=getattr(Config, "OPT_WEIGHT_DECAY", 0.0),
    )
    
    # 与 ROOT 对齐：ReduceLROnPlateau scheduler（参数对齐 ROOT_new/configs/Dkitty.yaml）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        verbose=True,
        threshold_mode='rel',
        cooldown=getattr(Config, "LR_SCHEDULER_COOLDOWN", 200),
        factor=getattr(Config, 'LR_SCHEDULER_FACTOR', 0.5),
        patience=getattr(Config, 'LR_SCHEDULER_PATIENCE', 200),
        threshold=getattr(Config, "LR_SCHEDULER_THRESHOLD", 1e-4),
        min_lr=getattr(Config, 'LR_SCHEDULER_MIN_LR', 5e-7)
    )
    
    # 与 ROOT 对齐：EMA（shadow/apply/restore + update_ema_interval + start_ema_step）
    use_ema = getattr(Config, "USE_EMA", True)
    ema = EMA(getattr(Config, "EMA_DECAY", 0.995)) if use_ema else None
    if ema is not None:
        ema.register(net)
    update_ema_interval = getattr(Config, "EMA_UPDATE_INTERVAL", 8)
    start_ema_step = getattr(Config, "EMA_START_STEP", 4000)

    # --- 与 ROOT BaseRunner 对齐：每个 Epoch 动态采样 GP + DataLoader + accumulate + per-step scheduler + EMA ---
    print(f"=== Training: Dynamic GP Sampling ({Config.FM_EPOCHS} Epochs) ===")
    print(f"每个 Epoch 采样 n_e = {Config.GP_NUM_FUNCTIONS} 个 GP 函数")
    print(f"每个 GP 函数采样 {Config.GP_NUM_POINTS} 个配对")
    print(f"总计将生成约 {Config.GP_NUM_FUNCTIONS} × {Config.FM_EPOCHS} = {Config.GP_NUM_FUNCTIONS * Config.FM_EPOCHS} 个 GP 函数")

    global_step = 0
    accumulate_grad_batches = getattr(Config, "ACCUMULATE_GRAD_BATCHES", 2)
    val_dataset = []
    validation_interval = getattr(Config, "VALIDATION_INTERVAL", 20)

    for epoch in range(Config.FM_EPOCHS):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch+1}/{Config.FM_EPOCHS} ===")

        # GP 采样输入：与 ROOT 一致用固定 best_x（top 1024）
        if getattr(Config, "GP_USE_FIXED_BEST_X", True) and Config.GP_TYPE_INITIAL_POINTS == "highest":
            gp_sampling_x = best_x
        else:
            # fallback（ROOT 没有，保留但通常不走）
            num_high = int(Config.GP_NUM_POINTS // 2)
            idx_high = top_k_indices[torch.randperm(len(top_k_indices))[:num_high]]
            num_rand = Config.GP_NUM_POINTS - num_high
            idx_rand = all_indices[torch.randperm(len(all_indices))[:num_rand]]
            mixed_indices = torch.cat([idx_high, idx_rand])
            gp_sampling_x = X_train_tensor[mixed_indices]

        # 构建 GP 模型（TFBind8 用部分样本）
        if Config.TASK_NAME == "TFBind8-Exact-v0":
            selected_fit_samples = torch.randperm(X_train_tensor.shape[0])[:Config.GP_NUM_FIT_SAMPLES]
            GP_Model = GP(
                device=device,
                x_train=X_train_tensor[selected_fit_samples],
                y_train=y_train_tensor[selected_fit_samples].view(-1),
                lengthscale=lengthscale,
                variance=variance,
                noise=noise,
                mean_prior=mean_prior,
            )
        else:
            GP_Model = GP(
                device=device,
                x_train=X_train_tensor,
                y_train=y_train_tensor.view(-1),
                lengthscale=lengthscale,
                variance=variance,
                noise=noise,
                mean_prior=mean_prior,
            )

        sampling_start = time.time()
        data_from_GP = sampling_data_from_GP(
            x_train=gp_sampling_x,
            device=device,
            GP_Model=GP_Model,
            num_functions=Config.GP_NUM_FUNCTIONS,
            num_gradient_steps=Config.GP_NUM_GRADIENT_STEPS,
            num_points=Config.GP_NUM_POINTS,
            learning_rate=Config.GP_LEARNING_RATE,
            delta_lengthscale=Config.GP_DELTA_LENGTHSCALE,
            delta_variance=Config.GP_DELTA_VARIANCE,
            seed=epoch,
            threshold_diff=Config.GP_THRESHOLD_DIFF,
            verbose=False,
        )
        sampling_time = time.time() - sampling_start

        # 与 ROOT 对齐：create_train_dataloader(val_frac=0.1, batch_size=64)
        val_frac = getattr(Config, "VAL_FRAC", 0.1)
        batch_size = Config.FM_BATCH_SIZE
        train_loader, current_epoch_val_dataset = create_train_dataloader(
            data_from_GP=data_from_GP,
            val_frac=val_frac,
            batch_size=batch_size,
            shuffle=True,
        )
        val_dataset = val_dataset + current_epoch_val_dataset

        if len(train_loader.dataset) == 0:
            print(f"Warning: No valid training data generated in epoch {epoch+1}")
            continue

        print(f"Created DataLoader: {len(train_loader.dataset)} train, {len(current_epoch_val_dataset)} val (epoch)")
        print(f"  [⏱️ Time] GP采样: {sampling_time:.2f}s")

        # ==============================
        #   训练阶段：根据 MODEL_TYPE 分支
        #   - BB: 走原有 Brownian Bridge 训练逻辑
        #   - FM: 使用端点对构造直线轨迹，调用 train_cfm_step
        # ==============================
        if use_bb:
            # ---------- Brownian Bridge 训练 ----------
            use_cfg = getattr(Config, "USE_CFG_TRAINING", False)
            cfg_prob = getattr(Config, "CFG_PROB", 0.15)

            net.train()
            optimizer.zero_grad()
            loss_sum = 0.0
            num_loss = 0

            pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01, disable=False)
            for (x_high, y_high), (x_low, y_low) in pbar:
                global_step += 1
                x_high = x_high.to(device)
                x_low = x_low.to(device)
                y_high = y_high.to(device)
                y_low = y_low.to(device)

                if use_cfg:
                    torch.manual_seed(global_step)
                    rand_mask = torch.rand(y_high.size(), device=y_high.device)
                    mask = rand_mask <= cfg_prob
                    y_high = y_high.clone()
                    y_low = y_low.clone()
                    y_high[mask] = 0.0
                    y_low[mask] = 0.0

                loss, _ = net(x_high, y_high, x_low, y_low)
                loss.backward()

                if global_step % accumulate_grad_batches == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step(loss)

                if ema is not None and global_step % (update_ema_interval * accumulate_grad_batches) == 0:
                    with_decay = False if global_step < start_ema_step else True
                    ema.update(net, with_decay=with_decay)

                loss_sum += float(loss.detach().mean().item())
                num_loss += 1
                pbar.set_description(f"Epoch: [{epoch + 1} / {Config.FM_EPOCHS}] iter: {global_step} loss: {loss.detach().mean().item():.4f}")

            avg_loss = loss_sum / max(1, num_loss)
            epoch_total_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{Config.FM_EPOCHS} | BB Loss: {avg_loss:.4f} | time: {epoch_total_time:.2f}s")

            # validation（与 ROOT 一致：validation_interval/最后一轮，且 val 时 apply EMA）
            if (epoch + 1) % validation_interval == 0 or (epoch + 1) == Config.FM_EPOCHS:
                val_loader = create_val_dataloader(val_dataset=val_dataset, batch_size=batch_size, shuffle=False)
                if ema is not None:
                    ema.apply_shadow(net)
                net.eval()
                with torch.no_grad():
                    val_loss_sum = 0.0
                    val_n = 0
                    for (vx_high, vy_high), (vx_low, vy_low) in val_loader:
                        vx_high = vx_high.to(device)
                        vx_low = vx_low.to(device)
                        vy_high = vy_high.to(device)
                        vy_low = vy_low.to(device)
                        vloss, _ = net(vx_high, vy_high, vx_low, vy_low)
                        val_loss_sum += float(vloss.detach().mean().item())
                        val_n += 1
                avg_val_loss = val_loss_sum / max(1, val_n)
                if ema is not None:
                    ema.restore(net)
                print(f"[Val] Epoch {epoch+1} | BB avg_val_loss={avg_val_loss:.4f} | batches={val_n}")

            # 与 ROOT 对齐：按 save_interval 保存 checkpoint（BB）
            save_interval = getattr(Config, "SAVE_INTERVAL", 20)
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == Config.FM_EPOCHS:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/bbdm_epoch_{epoch+1}.pt"
                ckpt = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if ema is not None:
                    ckpt["ema"] = ema.shadow
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved to {ckpt_path}")

        else:
            # ---------- Flow Matching 训练（与 BB 一致：条件 y_low, y_high）----------
            # 使用 GP 端点对构造轨迹 [x_low, x_high]，并收集对应的 y_low, y_high
            net.train()
            traj_list = []
            y_low_list = []
            y_high_list = []
            for (x_high, y_high), (x_low, y_low) in train_loader:
                x_high = x_high.to(device)
                x_low = x_low.to(device)
                traj_batch = torch.stack([x_low, x_high], dim=1)
                traj_list.append(traj_batch.cpu().numpy())
                # 与 BB 一致：每条轨迹对应一对 (y_low, y_high)；先转 CPU 再 numpy 避免 CUDA tensor 报错
                y_low_list.append((y_low.detach().cpu().numpy() if torch.is_tensor(y_low) else np.asarray(y_low)).reshape(-1))
                y_high_list.append((y_high.detach().cpu().numpy() if torch.is_tensor(y_high) else np.asarray(y_high)).reshape(-1))

            trajectories = np.concatenate(traj_list, axis=0)
            y_low_arr = np.concatenate(y_low_list, axis=0)
            y_high_arr = np.concatenate(y_high_list, axis=0)
            fm_loss, fm_cos_sim, fm_l_grad, fm_l_sigma = train_cfm_step(
                net,
                trajectories=trajectories,
                y_low=y_low_arr,
                y_high=y_high_arr,
                optimizer=optimizer,
                device=device,
                gp_model=None,
                weights=None,
            )

            if scheduler is not None:
                scheduler.step(fm_loss)

            # 简单的 EMA：按 epoch 级别更新
            if ema is not None:
                ema.update(net, with_decay=True)

            epoch_total_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{Config.FM_EPOCHS} | FM Loss: {fm_loss:.4f} | time: {epoch_total_time:.2f}s")

            # Flow Matching 分支暂不做额外的 val-loop，直接使用 loss 监控

            # 保存 Flow Matching checkpoint
            save_interval = getattr(Config, "SAVE_INTERVAL", 20)
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == Config.FM_EPOCHS:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/cfm_epoch_{epoch+1}.pt"
                ckpt = {
                    "epoch": epoch + 1,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if ema is not None:
                    ckpt["ema"] = ema.shadow
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved to {ckpt_path}")

    # 保存最终模型
    final_model_path = f"checkpoints/{'bb' if use_bb else 'cfm'}_model_final.pt"
    os.makedirs("checkpoints", exist_ok=True)
    final_ckpt = {
        'epoch': Config.FM_EPOCHS,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_type': 'BB' if use_bb else 'FM',
        'input_dim': input_dim,
        'hidden_dim': Config.HIDDEN_DIM,
    }
    if use_bb:
        final_ckpt.update({
            'BB_NUM_TIMESTEPS': Config.BB_NUM_TIMESTEPS,
            'BB_SAMPLE_STEP': Config.BB_SAMPLE_STEP,
            'BB_OBJECTIVE': Config.BB_OBJECTIVE,
        })
    torch.save(final_ckpt, final_model_path)
    print(f"\n[💾 Final Model] Saved to {final_model_path}")

    # 与 ROOT 对齐：测试前应用 EMA
    if ema is not None:
        print("\n[EMA] Applying EMA weights for evaluation...")
        ema.apply_shadow(net)
    
    # 4. 推理与 SOTA 评估 (Q=128)（与 ROOT 的测试逻辑对齐）
    eval_type = getattr(Config, 'EVAL_TYPE_SAMPLING', 'highest').lower()
    test_q = Config.NUM_TEST_SAMPLES
    sorted_indices = np.argsort(y_train_norm)

    if eval_type == 'low':
        # 与 ROOT 一致：从最低 percentile_sampling 比例中随机抽 test_q 个（ROOT sampling_from_offline_data）
        pct = getattr(Config, 'EVAL_PERCENTILE_SAMPLING', 0.2)
        n_low = int(pct * len(sorted_indices))  # 最低 20% 的个数
        low_pool = sorted_indices[:n_low]
        np.random.seed(Config.SEED)
        size = min(test_q, len(low_pool))
        chosen = np.random.choice(len(low_pool), size=size, replace=False)
        start_indices = low_pool[chosen]
        print(f"\n=== SOTA Evaluation (Low-point, Q=128, in-distribution) ===")
        print(f"Selected {size} from lowest {pct*100:.0f}% (n_low={len(low_pool)}, conditioning = low->high)")
    else:
        # 从最高 128 出发：与 ROOT type_sampling=highest 一致
        start_indices = sorted_indices[-test_q:]
        print(f"\n=== SOTA Evaluation (Highest-point, Q=128) ===")
        print(f"Selected {test_q} highest samples as starting points")

    test_q = len(start_indices)  # 实际评估数量（low 时可能 < 128）
    X_test_norm = X_train_norm[start_indices]
    y_test_start = y_train_norm[start_indices]
    print(f"Starting scores (normalized): mean={np.mean(y_test_start):.4f}, max={np.max(y_test_start):.4f}, min={np.min(y_test_start):.4f}")

    # 与 ROOT 完全一致：目标分数 = (oracle_y_max - mean_y)/std_y * alpha
    task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067, 'AntMorphology-Exact-v0': 590.24445, 'DKittyMorphology-Exact-v0': 340.90985}
    oracle_y_max = task_to_max[Config.TASK_NAME]
    normalized_oracle_y_max = (oracle_y_max - mean_y) / std_y
    test_alpha = getattr(Config, 'TEST_ALPHA', 0.8)
    y_high_cond_root = float(normalized_oracle_y_max * test_alpha)
    y_high_cond = np.full_like(y_test_start, y_high_cond_root)
    print(f"[ROOT-aligned] y_high_cond = (oracle_y_max - mean_y)/std_y * alpha = {y_high_cond_root:.4f} (alpha={test_alpha})")

    # 推理：FM 与 BB 一致，均条件 (y_low, y_high)，可从 top-128 出发往目标 y_high 走
    if use_bb:
        cfg_weight = getattr(Config, "BB_CFG_WEIGHT", -1.5) if getattr(Config, "BB_USE_CFG_TEST", False) else 0.0
        opt_X_norm = inference_bb(
            net, X_test_norm, y_test_start, y_high_cond, device,
            clip_denoised=getattr(Config, "BB_CLIP_DENOISED", False),
            cfg_weight=cfg_weight,
        )
    else:
        opt_X_norm = inference_ode(net, X_test_norm, y_test_start, y_high_cond, device)
    
    # 反标准化（与 ROOT 一致）
    opt_X_denorm = opt_X_norm * std_x + mean_x
    
    # 还原形状供 task.predict 打分（与 ROOT 一致）
    if task.is_discrete and logits_shape is not None:
        # 离散任务：需要 reshape 回 (N, L, V-1) 的形状
        # 使用数据加载时保存的 logits_shape 信息
        opt_X_for_predict = opt_X_denorm.reshape(test_q, logits_shape[1], logits_shape[2])
        # 原始样本也需要相同处理
        original_X_denorm = X_test_norm * std_x + mean_x
        original_X_for_predict = original_X_denorm.reshape(test_q, logits_shape[1], logits_shape[2])
        
        print(f"[Discrete Task] Reshaped to Logits format: {opt_X_for_predict.shape}")
    else:
        # 连续任务：直接使用
        opt_X_for_predict = opt_X_denorm
        original_X_for_predict = X_test_norm * std_x + mean_x
    
    # 使用 Oracle 评估（与 ROOT 一致，直接传入 numpy array）
    final_scores = task.predict(opt_X_for_predict).flatten()
    original_scores = task.predict(original_X_for_predict).flatten()
    
    # 与 ROOT 完全对齐：使用相同的 task_to_min, task_to_max, task_to_best 字典
    task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268, 'AntMorphology-Exact-v0': -386.90036, 'DKittyMorphology-Exact-v0': -880.4585}
    task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067, 'AntMorphology-Exact-v0': 590.24445, 'DKittyMorphology-Exact-v0': 340.90985}
    task_to_best = {'TFBind8-Exact-v0': 0.43929616, 'TFBind10-Exact-v0': 0.005328223, 'AntMorphology-Exact-v0': 165.32648, 'DKittyMorphology-Exact-v0': 199.36252}
    
    # 获取任务的 min 和 max 值（与 ROOT 一致）
    oracle_y_min = task_to_min[Config.TASK_NAME]
    oracle_y_max = task_to_max[Config.TASK_NAME]
    
    # 计算标准化分数（与 ROOT 完全一致）
    # ROOT 使用公式: final_score = (high_true_scores - oracle_y_min) / (oracle_y_max - oracle_y_min)
    final_score_normalized = (torch.from_numpy(final_scores) - oracle_y_min) / (oracle_y_max - oracle_y_min)
    original_score_normalized = (torch.from_numpy(original_scores) - oracle_y_min) / (oracle_y_max - oracle_y_min)
    
    # 计算百分位数（与 ROOT 完全一致：使用标准化后的分数）
    percentiles = torch.quantile(final_score_normalized, torch.tensor([1.0, 0.8, 0.5]), interpolation='higher')
    p100_score = percentiles[0].item()
    p80_score = percentiles[1].item()
    p50_score = percentiles[2].item()
    
    # 打印原始分数分布（用于调试）
    final_scores_sorted = np.sort(final_scores)
    print(f"\n[Result] Final scores distribution (raw):")
    print(f"  Min: {final_scores_sorted[0]:.4f}")
    print(f"  Max: {final_scores_sorted[-1]:.4f}")
    print(f"  Mean: {np.mean(final_scores):.4f}")
    print(f"  Std: {np.std(final_scores):.4f}")
    
    # 打印标准化后的分数分布
    final_score_normalized_np = final_score_normalized.numpy()
    print(f"\n[Result] Normalized scores distribution (comparable with ROOT):")
    print(f"  Min: {np.min(final_score_normalized_np):.4f}")
    print(f"  Max: {np.max(final_score_normalized_np):.4f}")
    print(f"  Mean: {np.mean(final_score_normalized_np):.4f}")
    print(f"  Std: {np.std(final_score_normalized_np):.4f}")
    
    print("-" * 60)
    print(f"Original Mean (Top {test_q}, raw): {np.mean(original_scores):.4f}")
    print(f"Optimized Mean (Top {test_q}, raw): {np.mean(final_scores):.4f}")
    print(f"Improvement (raw):                   {np.mean(final_scores) - np.mean(original_scores):.4f}")
    print("-" * 60)
    print(f"Normalized 100th Percentile (Max):      {p100_score:.6f}")
    print(f"Normalized 80th Percentile:             {p80_score:.6f}")
    print(f"Normalized 50th Percentile (Median):    {p50_score:.6f}")
    print("-" * 60)
    print(f"[ROOT Comparable] These normalized percentiles are directly comparable with ROOT results")

if __name__ == "__main__":
    main()


#TF10 0.685  0.526  0.473
#Ant 0.965 0.847 0.712
#Dkitty 0.972 0.938 0.919
#TF8  0.986   0.839 0.675