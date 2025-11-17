import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random  # 新增

# =========================
# 模型
# =========================
class RVAESeq2Seq(nn.Module):
    def __init__(self, x_dim=4, y_dim=1, h_dim=128, z_dim=16, bidirectional=True):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.bi = 2 if bidirectional else 1

        self.enc_gru = nn.GRU(input_size=x_dim, hidden_size=h_dim,
                              batch_first=True, bidirectional=bidirectional)
        self.enc_mu     = nn.Linear(self.bi * h_dim, z_dim)
        self.enc_logvar = nn.Linear(self.bi * h_dim, z_dim)

        self.dec_gru = nn.GRU(input_size=x_dim + z_dim, hidden_size=h_dim,
                              batch_first=True, bidirectional=bidirectional)
        self.dec_out = nn.Linear(self.bi * h_dim, y_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h, _ = self.enc_gru(x)
        mu     = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def decode(self, x, z):
        dec_in = torch.cat([x, z], dim=-1)
        h, _ = self.dec_gru(dec_in)
        y_hat = self.dec_out(h)
        return y_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decode(x, z)
        return y_hat, mu, logvar

# =========================
# 损失（含统计项）
# =========================
def elbo_loss(y_hat, y, mu, logvar, beta=1.0, reduction='mean',
              alpha_max=0.15, alpha_min=0.2, alpha_mean=0.05):
    rec = F.mse_loss(y_hat, y, reduction='none')  # (B,T,C)
    rec = rec.mean(dim=-1)                        # (B,T)

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B,T,z)
    kl = kl.sum(dim=-1)                                   # (B,T)

    # 统计特征
    y_hat_max = y_hat.max(dim=1).values
    y_max     = y.max(dim=1).values
    y_hat_min = y_hat.min(dim=1).values
    y_min     = y.min(dim=1).values
    y_hat_mean = y_hat.mean(dim=1)
    y_mean     = y.mean(dim=1)

    l_max  = F.mse_loss(y_hat_max,  y_max,  reduction='none').mean(dim=-1)
    l_min  = F.mse_loss(y_hat_min,  y_min,  reduction='none').mean(dim=-1)
    l_mean = F.mse_loss(y_hat_mean, y_mean, reduction='none').mean(dim=-1)
    stats = alpha_max * l_max + alpha_min * l_min + alpha_mean * l_mean  # (B,)

    if reduction == 'mean':
        rec = rec.mean()
        kl  = kl.mean()
        stats = stats.mean()
    elif reduction == 'sum':
        rec = rec.sum()
        kl  = kl.sum()
        stats = stats.sum()

    total = rec + stats + beta * kl
    return total, rec, kl, stats

@torch.no_grad()
def predict_mean(model, x):
    model.eval()
    mu, logvar = model.encode(x)
    y_hat = model.decode(x, mu)
    return y_hat

# =========================
# 归一化工具
# =========================
def compute_channel_norm_params(X):
    eps = 1e-8
    mu  = X.mean(axis=(0,1))
    std = X.std(axis=(0,1), ddof=0) + eps
    return mu, std

def apply_channel_norm(X, mu, std):
    return (X - mu.reshape(1,1,-1)) / std.reshape(1,1,-1)

def inverse_channel_norm(X_norm, mu, std):
    return X_norm * std.reshape(1,1,-1) + mu.reshape(1,1,-1)

def save_norm_params_npz(path, x_mu, x_std, y_mu, y_std):
    np.savez(path, x_mu=x_mu, x_std=x_std, y_mu=y_mu, y_std=y_std)

# =========================
# 评估（原始尺度）
# =========================
def evaluate_original_scale(model, loader, y_mu, y_std, device):
    model.eval()
    se_sum = ae_sum = 0.0
    n_elems = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for xb, yb_n in loader:
            xb = xb.to(device)
            yb_n = yb_n.to(device)
            y_pred_n = predict_mean(model, xb).cpu().numpy()
            y_true_n = yb_n.cpu().numpy()
            y_pred = inverse_channel_norm(y_pred_n, y_mu, y_std)
            y_true = inverse_channel_norm(y_true_n, y_mu, y_std)
            se_sum += ((y_pred - y_true) ** 2).sum()
            ae_sum += np.abs(y_pred - y_true).sum()
            n_elems += y_true.size
            all_preds.append(y_pred.reshape(-1))
            all_trues.append(y_true.reshape(-1))
    mse  = se_sum / n_elems
    rmse = float(np.sqrt(mse))
    mae  = ae_sum / n_elems
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    pcc, p_value = pearsonr(all_preds, all_trues)
    return mse, rmse, mae, pcc, p_value

# =========================
# 单折训练
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_fold(device,
                   X_tr_fold, Y_tr_fold,
                   X_val_fold, Y_val_fold,
                   alpha_max=0.15, alpha_min=0.2, alpha_mean=0.05,
                   bs=32, epochs=300, warmup=10, patience=20, fold_dir=".",
                   seed=42):  # 新增 seed
    os.makedirs(fold_dir, exist_ok=True)
    # 创建 DataLoader 生成器
    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)

    # 仅训练折做归一化
    x_mu, x_std = compute_channel_norm_params(X_tr_fold)
    y_mu, y_std = compute_channel_norm_params(Y_tr_fold)
    save_norm_params_npz(os.path.join(fold_dir, "norm_params_fold_train.npz"),
                         x_mu, x_std, y_mu, y_std)

    # 归一化
    X_tr_n = apply_channel_norm(X_tr_fold, x_mu, x_std)
    Y_tr_n = apply_channel_norm(Y_tr_fold, y_mu, y_std)
    X_val_n = apply_channel_norm(X_val_fold, x_mu, x_std)
    Y_val_n = apply_channel_norm(Y_val_fold, y_mu, y_std)

    # Tensor
    X_tr_t = torch.from_numpy(X_tr_n).float()
    Y_tr_t = torch.from_numpy(Y_tr_n).float()
    X_val_t = torch.from_numpy(X_val_n).float()
    Y_val_t = torch.from_numpy(Y_val_n).float()

    train_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t),
                              batch_size=bs, shuffle=True, drop_last=True,
                              generator=dl_gen)  # 传入生成器
    val_loader   = DataLoader(TensorDataset(X_val_t, Y_val_t),
                              batch_size=bs, shuffle=False, drop_last=False)

    model = RVAESeq2Seq(x_dim=4, y_dim=1, h_dim=128, z_dim=16, bidirectional=True).to(device)
    optim_ = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_rmse = float('inf')
    best_epoch = -1
    patience_cnt = 0
    best_path = os.path.join(fold_dir, "best_model.pt")

    for ep in range(epochs):
        model.train()
        beta = min(1.0, (ep + 1) / warmup)
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_hat, mu, logvar = model(xb)
            loss, rec, kl, stats = elbo_loss(
                y_hat, yb, mu, logvar,
                beta=beta, reduction='mean',
                alpha_max=alpha_max, alpha_min=alpha_min, alpha_mean=alpha_mean
            )
            optim_.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()

        # 验证 RMSE（原始尺度）
        _, val_rmse, val_mae, _, _ = evaluate_original_scale(model, val_loader, y_mu, y_std, device)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = ep
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    val_mse, val_rmse, val_mae, _, _ = evaluate_original_scale(model, val_loader, y_mu, y_std, device)

    # 释放
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_mse": float(val_mse),
        "val_rmse": float(val_rmse),
        "val_mae": float(val_mae),
        "best_epoch": int(best_epoch)
    }

# =========================
# 最终模型训练（在合并 train+val 上再划分少量验证）
# =========================
def extract_subject_id(fname: str):
    """
    按用户说明：前面 TRM172-RHC1 相同视为同一个人。
    采用在 '_seg' 之前的部分作为受试者 ID。
    若没有 '_seg'，则取第一个 '_' 前的段。
    """
    base = os.path.basename(fname.strip())
    if "_seg" in base:
        return base.split("_seg")[0]
    parts = base.split("_")
    return parts[0]

def load_filenames_list(base_dir, split_name):
    """
    读取 filenames_{split_name}.txt (如 filenames_train.txt)
    返回列表，若文件不存在返回 None。
    """
    path = os.path.join(base_dir, f"filenames_{split_name}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

def load_subject_sources(base_dir, csv_name="train_list.csv", source_col="source"):
    """
    从 CSV 读取与 train npy 一一对应的受试者来源列（如 TRM197-RHC1）。
    """
    path = os.path.join(base_dir, csv_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or source_col not in reader.fieldnames:
            raise ValueError(f"CSV 缺少列 '{source_col}': {path}")
        sources = [ (row[source_col] or "").strip() for row in reader ]
    return sources

def train_final_and_test(device,
                         X_all, Y_all,
                         X_te, Y_te,
                         alpha_max=0.15, alpha_min=0.2, alpha_mean=0.05,
                         bs=32, epochs=300, warmup=10, patience=20,
                         val_ratio=0.1, out_dir="final_model_result",
                         refit_on_all=False,
                         groups_all=None,
                         seed=42):  # 新增 seed
    os.makedirs(out_dir, exist_ok=True)

    N = X_all.shape[0]
    if groups_all is not None and len(groups_all) == N:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        tr_idx, va_idx = next(gss.split(np.arange(N), groups=groups_all))
        print(f"[Final Split] 使用 GroupShuffleSplit 按受试者划分: train={len(tr_idx)}, val={len(va_idx)}")
    else:
        rng = np.random.default_rng(seed)  # 使用统一 seed
        N_val = max(1, int(N * val_ratio))
        val_indices = rng.choice(N, size=N_val, replace=False)
        train_mask = np.ones(N, dtype=bool)
        train_mask[val_indices] = False
        tr_idx = np.where(train_mask)[0]
        va_idx = val_indices
        print(f"[Final Split] 使用随机划分(seed={seed}): train={len(tr_idx)}, val={len(va_idx)}")

    np.savez(os.path.join(out_dir, "final_train_val_indices.npz"),
             train_idx=tr_idx, val_idx=va_idx, seed=seed)

    X_tr = X_all[tr_idx]; Y_tr = Y_all[tr_idx]
    X_val = X_all[va_idx]; Y_val = Y_all[va_idx]

    x_mu, x_std = compute_channel_norm_params(X_tr)
    y_mu, y_std = compute_channel_norm_params(Y_tr)
    save_norm_params_npz(os.path.join(out_dir, "norm_params_train_only.npz"),
                         x_mu, x_std, y_mu, y_std)

    X_tr_n  = apply_channel_norm(X_tr,  x_mu, x_std)
    Y_tr_n  = apply_channel_norm(Y_tr,  y_mu, y_std)
    X_val_n = apply_channel_norm(X_val, x_mu, x_std)
    Y_val_n = apply_channel_norm(Y_val, y_mu, y_std)
    X_te_n  = apply_channel_norm(X_te,  x_mu, x_std)
    Y_te_n  = apply_channel_norm(Y_te,  y_mu, y_std)

    dl_gen = torch.Generator(); dl_gen.manual_seed(seed)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr_n).float(),
                                            torch.from_numpy(Y_tr_n).float()),
                              batch_size=bs, shuffle=True, drop_last=True,
                              generator=dl_gen)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val_n).float(),
                                            torch.from_numpy(Y_val_n).float()),
                              batch_size=bs, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_te_n).float(),
                                            torch.from_numpy(Y_te_n).float()),
                              batch_size=bs, shuffle=False, drop_last=False)

    model = RVAESeq2Seq(x_dim=4, y_dim=1, h_dim=128, z_dim=16, bidirectional=True).to(device)
    optim_ = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_rmse = float('inf'); best_epoch = -1; patience_cnt = 0
    best_path = os.path.join(out_dir, "best_model.pt")

    for ep in range(epochs):
        model.train()
        beta = min(1.0, (ep + 1) / warmup)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_hat, mu, logvar = model(xb)
            loss, rec, kl, stats = elbo_loss(
                y_hat, yb, mu, logvar,
                beta=beta, reduction='mean',
                alpha_max=alpha_max, alpha_min=alpha_min, alpha_mean=alpha_mean
            )
            optim_.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
        _, val_rmse, _, _, _ = evaluate_original_scale(model, val_loader, y_mu, y_std, device)
        print(f"[Final Train] Epoch {ep+1:03d} | Val RMSE={val_rmse:.4f} | beta={beta:.2f}")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = ep
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[Final EarlyStop] best epoch={best_epoch+1}, best val RMSE={best_val_rmse:.4f}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    if refit_on_all:
        print(f"[Refit] 使用 train+val 全量数据重新训练 {best_epoch+1} 轮 (seed={seed})")
        x_mu_all, x_std_all = compute_channel_norm_params(X_all)
        y_mu_all, y_std_all = compute_channel_norm_params(Y_all)
        save_norm_params_npz(os.path.join(out_dir, "norm_params_trainval_all.npz"),
                             x_mu_all, x_std_all, y_mu_all, y_std_all)

        X_all_n = apply_channel_norm(X_all, x_mu_all, x_std_all)
        Y_all_n = apply_channel_norm(Y_all, y_mu_all, y_std_all)
        all_loader = DataLoader(TensorDataset(torch.from_numpy(X_all_n).float(),
                                              torch.from_numpy(Y_all_n).float()),
                                batch_size=bs, shuffle=True, drop_last=True,
                                generator=dl_gen)
        model = RVAESeq2Seq(x_dim=4, y_dim=1, h_dim=128, z_dim=16, bidirectional=True).to(device)
        optim_refit = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(best_epoch + 1):
            model.train()
            beta = min(1.0, (ep + 1) / warmup)
            for xb, yb in all_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_hat, mu, logvar = model(xb)
                loss, rec, kl, stats = elbo_loss(
                    y_hat, yb, mu, logvar,
                    beta=beta, reduction='mean',
                    alpha_max=alpha_max, alpha_min=alpha_min, alpha_mean=alpha_mean
                )
                optim_refit.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim_refit.step()
        y_mu, y_std = y_mu_all, y_std_all
        X_te_n = apply_channel_norm(X_te, x_mu_all, x_std_all)
        Y_te_n = apply_channel_norm(Y_te, y_mu_all, y_std_all)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te_n).float(),
                                               torch.from_numpy(Y_te_n).float()),
                                 batch_size=bs, shuffle=False, drop_last=False)

    test_mse, test_rmse, test_mae, test_pcc, p_value = evaluate_original_scale(model, test_loader, y_mu, y_std, device)
    print(f"[TEST original] MSE={test_mse:.6f} | RMSE={test_rmse:.6f} | MAE={test_mae:.6f} | PCC={test_pcc:.6f} | p-value={p_value:.6f}")

    # 保存预测
    all_y_pred, all_y_true = [], []
    with torch.no_grad():
        for xb, yb_n in test_loader:
            xb, yb_n = xb.to(device), yb_n.to(device)
            y_pred_n = predict_mean(model, xb).cpu().numpy()
            y_true_n = yb_n.cpu().numpy()
            all_y_pred.append(y_pred_n)
            all_y_true.append(y_true_n)
    y_pred = np.concatenate(all_y_pred, axis=0)
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = inverse_channel_norm(y_pred, y_mu, y_std)
    y_true = inverse_channel_norm(y_true, y_mu, y_std)
    np.savez_compressed(os.path.join(out_dir, "test_y_true_y_pred_original_scale.npz"),
                        y_true=y_true.astype(np.float32),
                        y_pred=y_pred.astype(np.float32))
    print("[SAVE] 测试集 y_true / y_pred 原始尺度已保存。")

    # 绘图
    plot_dir = os.path.join(out_dir, "test_plots")
    os.makedirs(plot_dir, exist_ok=True)
    x1_arr = X_te[..., 0]; x2_arr = X_te[..., 1]; x3_arr = X_te[..., 2]; x4_arr = X_te[..., 3]
    y_arr = y_true[..., 0]; yhat_arr = y_pred[..., 0]
    N_test, T = y_arr.shape
    for i in range(N_test):
        fig, axes = plt.subplots(5, 1, figsize=(6, 10), sharex=True)
        t = np.arange(T)
        axes[0].plot(t, x1_arr[i]); axes[0].set_ylabel("ECG"); axes[0].set_title(f"Test Sample {i}")
        axes[1].plot(t, x2_arr[i]); axes[1].set_ylabel("SCG_DV")
        axes[2].plot(t, x3_arr[i]); axes[2].set_ylabel("SCG_HtoF")
        axes[3].plot(t, x4_arr[i]); axes[3].set_ylabel("SCG_Lat")
        axes[4].plot(t, y_arr[i], label="True")
        axes[4].plot(t, yhat_arr[i], label="Pred")
        axes[4].legend(); axes[4].set_ylabel("PA"); axes[4].set_xlabel("Time")
        fig.savefig(os.path.join(plot_dir, f"test_sample_{i}_fiveplots.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[PLOT] 已生成 {N_test} 个测试样本图像到 {plot_dir}")

    return {
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_pcc": test_pcc,
        "best_epoch": best_epoch,
        "out_dir": out_dir,
        "refit": refit_on_all
    }

# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio_final", type=float, default=0.1)
    parser.add_argument("--base_data_dir", type=str,
                        default=r"E:\Cardiovascular\CR-VAE-main\exports1109")
    parser.add_argument("--result_root", type=str,
                        default="result_fixed_alpha_5fold_1112_0-0-0")
    parser.add_argument("--refit_all", type=int, default=1)
    parser.add_argument("--group_fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="统一随机种子")  # 新增
    args = parser.parse_args()

    set_seed(args.seed)  # 设定随机种子

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > args.gpu
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU cuda:{args.gpu} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("Using CPU")

    os.makedirs(args.result_root, exist_ok=True)

    base = args.base_data_dir
    # 仅读取已合并后的 train 数据
    x1_tr = np.load(os.path.join(base, "ECG_train.npy"))
    x2_tr = np.load(os.path.join(base, "SCG_dv_train.npy"))
    x3_tr = np.load(os.path.join(base, "SCG_hf_train.npy"))
    x4_tr = np.load(os.path.join(base, "SCG_lat_train.npy"))
    y_tr  = np.load(os.path.join(base, "PA_train.npy"))
    X_tr = np.stack([x1_tr, x2_tr, x3_tr, x4_tr], axis=-1)
    Y_tr = y_tr[..., None]

    # 测试集保持不变
    x1_te = np.load(os.path.join(base, "ECG_test.npy"))
    x2_te = np.load(os.path.join(base, "SCG_dv_test.npy"))
    x3_te = np.load(os.path.join(base, "SCG_hf_test.npy"))
    x4_te = np.load(os.path.join(base, "SCG_lat_test.npy"))
    y_te  = np.load(os.path.join(base, "PA_test.npy"))
    X_te = np.stack([x1_te, x2_te, x3_te, x4_te], axis=-1)
    Y_te = y_te[..., None]

    # 现在的全量数据（原 train+val 已经离线合并成 train）
    X_all = X_tr
    Y_all = Y_tr
    N_all = X_all.shape[0]

    # 从 CSV 读取受试者来源（与 npy 行顺序一一对应）
    sources = load_subject_sources(base, csv_name="train_list.csv", source_col="source")
    groups_all = None
    if args.group_fold == 1 and sources is not None and len(sources) == N_all:
        groups_all = np.array(sources)
        print("[GroupKFold] 受试者 source 示例:", groups_all[:5])
        splitter = GroupKFold(n_splits=5)
        split_iter = splitter.split(np.arange(N_all), groups=groups_all)
        print("[5-Fold] 使用 GroupKFold（受试者不跨折）")
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = splitter.split(np.arange(N_all))
        print("[5-Fold] 使用普通 KFold")

    fold_results = []
    fixed_alpha = (0, 0, 0)
    print(f"[5-Fold] 固定 alpha_max={fixed_alpha[0]}, alpha_min={fixed_alpha[1]}, alpha_mean={fixed_alpha[2]}")

    for fold_idx, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        X_tr_fold = X_all[tr_idx]
        Y_tr_fold = Y_all[tr_idx]
        X_val_fold = X_all[va_idx]
        Y_val_fold = Y_all[va_idx]
        fold_dir = os.path.join(args.result_root, f"fold_{fold_idx}")
        res = train_one_fold(device,
                             X_tr_fold, Y_tr_fold,
                             X_val_fold, Y_val_fold,
                             alpha_max=fixed_alpha[0],
                             alpha_min=fixed_alpha[1],
                             alpha_mean=fixed_alpha[2],
                             bs=args.batch_size,
                             epochs=args.epochs,
                             warmup=args.warmup,
                             patience=args.patience,
                             fold_dir=fold_dir,
                             seed=args.seed)  # 传 seed
        fold_results.append(res)
        print(f"[Fold {fold_idx}] val_rmse={res['val_rmse']:.6f} | val_mae={res['val_mae']:.6f}")

    cv_csv = os.path.join(args.result_root, "cv_summary.csv")
    with open(cv_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "val_mse", "val_rmse", "val_mae", "best_epoch"])
        for i, r in enumerate(fold_results, start=1):
            w.writerow([i, r["val_mse"], r["val_rmse"], r["val_mae"], r["best_epoch"]])
        mean_mse = np.mean([r["val_mse"] for r in fold_results])
        mean_rmse = np.mean([r["val_rmse"] for r in fold_results])
        mean_mae = np.mean([r["val_mae"] for r in fold_results])
        w.writerow(["mean", mean_mse, mean_rmse, mean_mae, ""])
    print(f"[5-Fold] mean RMSE={mean_rmse:.6f} | mean MAE={mean_mae:.6f}")

    final_out_dir = os.path.join(args.result_root, "final_model")
    final_res = train_final_and_test(device,
                                     X_all, Y_all,
                                     X_te, Y_te,
                                     alpha_max=fixed_alpha[0],
                                     alpha_min=fixed_alpha[1],
                                     alpha_mean=fixed_alpha[2],
                                     bs=args.batch_size,
                                     epochs=args.epochs,
                                     warmup=args.warmup,
                                     patience=args.patience,
                                     val_ratio=args.val_ratio_final,
                                     out_dir=final_out_dir,
                                     refit_on_all=bool(args.refit_all),
                                     groups_all=groups_all,
                                     seed=args.seed)  # 传 seed
    print(f"[FINAL TEST] RMSE={final_res['test_rmse']:.6f} | MAE={final_res['test_mae']:.6f} | PCC={final_res['test_pcc']:.6f} | Refit={final_res['refit']}")

if __name__ == "__main__":
    main()