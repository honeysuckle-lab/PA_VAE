import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# =========================
# Model
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

def elbo_loss(y_hat, y, mu, logvar, beta=1.0, reduction='mean'):
    rec = F.mse_loss(y_hat, y, reduction='none')
    rec = rec.mean(dim=-1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl.sum(dim=-1)
    if reduction == 'mean':
        rec = rec.mean()
        kl  = kl.mean()
    elif reduction == 'sum':
        rec = rec.sum()
        kl  = kl.sum()
    return rec + beta * kl, rec, kl

@torch.no_grad()
def predict_mean(model, x):
    model.eval()
    mu, logvar = model.encode(x)
    y_hat = model.decode(x, mu)
    return y_hat

@torch.no_grad()
def evaluate_on_loader(model, loader, device):
    model.eval()
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        y_pred = predict_mean(model, xb)
        se_sum += F.mse_loss(y_pred, yb, reduction='sum').item()
        ae_sum += F.l1_loss(y_pred, yb, reduction='sum').item()
        n_elems += yb.numel()
    mse = se_sum / n_elems
    mae = ae_sum / n_elems
    rmse = np.sqrt(mse)
    return mse, rmse, mae

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

def load_norm_params_npz(path):
    f = np.load(path)
    return f['x_mu'], f['x_std'], f['y_mu'], f['y_std']

def make_kfold_indices(N, n_splits=5, shuffle=True, seed=42):
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    pairs = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        pairs.append((train_idx, val_idx))
    return pairs

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 5-Fold Cross Validation
# =========================
if __name__ == "__main__":
    set_seed(42)
    # 选择第6张卡（cuda:5）
    gpu_index = 5
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > gpu_index
    device = torch.device(f"cuda:{gpu_index}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(gpu_index)
        print(f"Using GPU cuda:{gpu_index} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print(f"Using CPU（请求的 GPU {gpu_index} 不可用或未可见）")

    # ========== 数据读取 ==========
    x1_tr = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\ECG_train.npy")
    x2_tr = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_dv_train.npy")
    x3_tr = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_hf_train.npy")
    x4_tr = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_lat_train.npy")
    y_tr  = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\PA_train.npy")

    X_tr = np.stack([x1_tr, x2_tr, x3_tr, x4_tr], axis=-1)   # (N_tr, T, 4)
    Y_tr = y_tr[..., None]                                    # (N_tr, T, 1)

    x1_te = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\ECG_test.npy")
    x2_te = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_dv_test.npy")
    x3_te = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_hf_test.npy")
    x4_te = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_lat_test.npy")
    y_te  = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\PA_test.npy")

    X_te = np.stack([x1_te, x2_te, x3_te, x4_te], axis=-1)   # (N_te, T, 4)
    Y_te = y_te[..., None]                                    # (N_te, T, 1)

    # ========== 5 折 ==========
    n_splits = 5
    pairs = make_kfold_indices(N=X_tr.shape[0], n_splits=n_splits, shuffle=True, seed=42)

    # 训练配置
    bs = 32
    epochs = 300
    warmup = 10
    patience = 20

    # 结果汇总
    test_mse_list, test_rmse_list, test_mae_list, test_pcc_list = [], [], [], []

    # 输出目录
    base_result_dir = "result_data_with_val_1018_5fold"
    os.makedirs(base_result_dir, exist_ok=True)

    # 控制绘图数量（为 0 则不绘图）
    max_plot_samples = 0

    for fold_idx, (tr_idx, val_idx) in enumerate(pairs, start=1):
        print(f"\n========== Fold {fold_idx}/{n_splits} ==========")
        # 拆分本折数据（使用原始尺度）
        X_tr_fold, Y_tr_fold = X_tr[tr_idx], Y_tr[tr_idx]
        X_val_fold, Y_val_fold = X_tr[val_idx], Y_tr[val_idx]

        # 每折：仅用训练划分计算归一化参数，避免泄漏
        x_mu, x_std = compute_channel_norm_params(X_tr_fold)
        y_mu, y_std = compute_channel_norm_params(Y_tr_fold)

        # 保存本折归一化参数
        fold_dir = os.path.join(base_result_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        save_norm_params_npz(os.path.join(fold_dir, f"rvae_norm_params_fold{fold_idx}.npz"),
                             x_mu, x_std, y_mu, y_std)

        # 应用归一化到 训练/验证/测试
        X_tr_n   = apply_channel_norm(X_tr_fold,  x_mu, x_std)
        Y_tr_n   = apply_channel_norm(Y_tr_fold,  y_mu, y_std)
        X_val_n  = apply_channel_norm(X_val_fold, x_mu, x_std)
        Y_val_n  = apply_channel_norm(Y_val_fold, y_mu, y_std)
        X_te_n   = apply_channel_norm(X_te,       x_mu, x_std)
        Y_te_n   = apply_channel_norm(Y_te,       y_mu, y_std)

        # 转 tensor
        X_tr_t  = torch.from_numpy(X_tr_n).float()
        Y_tr_t  = torch.from_numpy(Y_tr_n).float()
        X_val_t = torch.from_numpy(X_val_n).float()
        Y_val_t = torch.from_numpy(Y_val_n).float()
        X_te_t  = torch.from_numpy(X_te_n).float()
        Y_te_t  = torch.from_numpy(Y_te_n).float()

        # DataLoader
        train_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=bs, shuffle=True,  drop_last=True)
        val_loader   = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=bs, shuffle=False, drop_last=False)
        test_loader  = DataLoader(TensorDataset(X_te_t, Y_te_t),   batch_size=bs, shuffle=False, drop_last=False)

        # 模型与优化器
        model = RVAESeq2Seq(x_dim=4, y_dim=1, h_dim=128, z_dim=16, bidirectional=True).to(device)
        optim_ = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 训练（带早停）
        best_val_loss = float('inf')
        best_epoch = -1
        patience_counter = 0
        save_path = os.path.join(fold_dir, f"best_model_fold{fold_idx}.pt")

        for ep in range(epochs):
            model.train()
            total, total_rec, total_kl = 0.0, 0.0, 0.0
            beta = min(1.0, (ep + 1) / warmup)

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                y_hat, mu, logvar = model(xb)
                loss, rec, kl = elbo_loss(y_hat, yb, mu, logvar, beta=beta, reduction='mean')

                optim_.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim_.step()

                total     += loss.item()
                total_rec += rec.item()
                total_kl  += kl.item()

            # 验证评估（归一化尺度）
            val_loss, val_rec, val_kl = 0.0, 0.0, 0.0
            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    y_hat, mu, logvar = model(xb)
                    loss, rec, kl = elbo_loss(y_hat, yb, mu, logvar, beta=beta, reduction='mean')
                    val_loss += loss.item()
                    val_rec  += rec.item()
                    val_kl   += kl.item()
            val_loss /= len(val_loader)
            val_rec  /= len(val_loader)
            val_kl   /= len(val_loader)

            print(f"[Fold {fold_idx}] Epoch {ep+1:03d} | Train {total/len(train_loader):.4f} | Rec {total_rec/len(train_loader):.4f} | KL {total_kl/len(train_loader):.4f} | "
                  f"Val {val_loss:.4f} | Val Rec {val_rec:.4f} | Val KL {val_kl:.4f} | beta={beta:.2f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = ep
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Fold {fold_idx} Early Stop] 最佳epoch: {best_epoch+1}, 最佳验证损失: {best_val_loss:.4f}")
                    break

        # 加载最佳模型并在测试集（原始尺度）评估
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()

        se_sum = ae_sum = 0.0
        n_elems = 0
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for xb, yb_n in test_loader:
                xb = xb.to(device)
                yb_n = yb_n.to(device)
                y_pred_n = predict_mean(model, xb)

                y_pred = inverse_channel_norm(y_pred_n.cpu().numpy(), y_mu, y_std)
                y_true = inverse_channel_norm(yb_n.cpu().numpy(),   y_mu, y_std)

                se_sum += ((y_pred - y_true) ** 2).sum()
                ae_sum += np.abs(y_pred - y_true).sum()
                n_elems += y_true.size

                all_preds.append(y_pred.reshape(-1))
                all_trues.append(y_true.reshape(-1))

        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)

        mse  = se_sum / n_elems
        rmse = float(np.sqrt(mse))
        mae  = ae_sum / n_elems
        pcc, p_value = pearsonr(all_preds, all_trues)

        print(f"[Fold {fold_idx} TEST: original scale] MSE={mse:.6f} | RMSE={rmse:.6f} | MAE={mae:.6f} | PCC={pcc:.6f} | p-value={p_value:.6f}")

        test_mse_list.append(mse)
        test_rmse_list.append(rmse)
        test_mae_list.append(mae)
        test_pcc_list.append(pcc)

        # 保存本折预测
        # 同时可选绘图（默认关闭以避免大量图像）
        y_pred_all = []
        y_true_all = []
        with torch.no_grad():
            for xb, yb_n in test_loader:
                xb = xb.to(device)
                yb_n = yb_n.to(device)
                y_pred_n = predict_mean(model, xb).cpu().numpy()
                y_true_n = yb_n.cpu().numpy()
                y_pred_all.append(y_pred_n)
                y_true_all.append(y_true_n)
        y_pred_arr = np.concatenate(y_pred_all, axis=0)
        y_true_arr = np.concatenate(y_true_all, axis=0)
        y_pred_arr = inverse_channel_norm(y_pred_arr, y_mu, y_std)
        y_true_arr = inverse_channel_norm(y_true_arr, y_mu, y_std)

        save_npz_path = os.path.join(fold_dir, f"fold{fold_idx}_y_true_y_pred_original_scale.npz")
        np.savez_compressed(save_npz_path,
                            y_true=y_true_arr.astype(np.float32),
                            y_pred=y_pred_arr.astype(np.float32))
        print(f"[Fold {fold_idx} SAVE] 原始尺度 y_true/y_pred: {save_npz_path}")

        # 可选绘图
        if max_plot_samples > 0:
            x1_arr = X_te[..., 0]
            x2_arr = X_te[..., 1]
            x3_arr = X_te[..., 2]
            x4_arr = X_te[..., 3]
            y_arr  = y_true_arr[..., 0]
            yhat_arr = y_pred_arr[..., 0]

            N_test, T = y_arr.shape
            n_plot = min(max_plot_samples, N_test)
            for i in range(n_plot):
                fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
                t = np.arange(T)
                for ax in axes:
                    try:
                        ax.set_box_aspect(1)
                    except AttributeError:
                        ax.set_aspect('equal', adjustable='box')

                axes[0].plot(t, x1_arr[i], linewidth=1.2); axes[0].set_ylabel("x1")
                axes[0].set_title(f"Fold {fold_idx} - Test Sample {i} — Inputs & Residual")
                axes[1].plot(t, x2_arr[i], linewidth=1.2); axes[1].set_ylabel("x2")
                axes[2].plot(t, x3_arr[i], linewidth=1.2); axes[2].set_ylabel("x3")
                axes[3].plot(t, x4_arr[i], linewidth=1.2); axes[3].set_ylabel("x4")
                axes[4].plot(t, y_arr[i], label="True (denorm)")
                axes[4].plot(t, yhat_arr[i], label="Pred (denorm)")
                axes[4].legend()
                axes[4].set_title(f"Fold {fold_idx} - Test Sample {i} (original scale)")
                axes[4].set_xlabel("Time"); axes[4].set_ylabel("Target")

                fig.savefig(os.path.join(fold_dir, f"fold{fold_idx}_test_sample_{i}_fiveplots.png"),
                            dpi=300, bbox_inches="tight")
                plt.close(fig)

    # 汇总 5 折测试指标
    def mean_std(a):
        return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

    mse_mean, mse_std   = mean_std(test_mse_list)
    rmse_mean, rmse_std = mean_std(test_rmse_list)
    mae_mean, mae_std   = mean_std(test_mae_list)
    pcc_mean, pcc_std   = mean_std(test_pcc_list)

    print("\n========== 5-Fold Test Summary (original scale) ==========")
    print(f"MSE  mean±std: {mse_mean:.6f} ± {mse_std:.6f}")
    print(f"RMSE mean±std: {rmse_mean:.6f} ± {rmse_std:.6f}")
    print(f"MAE  mean±std: {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"PCC  mean±std: {pcc_mean:.6f} ± {pcc_std:.6f}")