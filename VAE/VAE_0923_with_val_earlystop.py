import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import os

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

# =========================
# Main with validation & early stopping
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 1) 读取训练集 ==========
    x1_tr = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\ECG_train.npy")
    x2_tr = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_dv_train.npy")
    x3_tr = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_hf_train.npy")
    x4_tr = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_lat_train.npy")
    y_tr  = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\PA_train.npy")

    X_tr = np.stack([x1_tr, x2_tr, x3_tr, x4_tr], axis=-1)
    Y_tr = y_tr[..., None]

    # ========== 2) 读取测试集 ==========
    x1_te = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\ECG_test.npy")
    x2_te = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_dv_test.npy")
    x3_te = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_hf_test.npy")
    x4_te = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\SCG_lat_test.npy")
    y_te  = np.load(r"E:\Cardiovascular\8-\VAE\CR-VAE-main\CR-VAE-main\data_500_filter\PA_test.npy")

    X_te = np.stack([x1_te, x2_te, x3_te, x4_te], axis=-1)
    Y_te = y_te[..., None]

    # ===== 仅用训练集计算归一化参数（按通道）并保存 =====
    x_mu, x_std = compute_channel_norm_params(X_tr)
    y_mu, y_std = compute_channel_norm_params(Y_tr)
    save_norm_params_npz('rvae_norm_params.npz', x_mu, x_std, y_mu, y_std)

    # ===== 用训练集参数对 训练/测试 做标准化 =====
    X_tr_n = apply_channel_norm(X_tr, x_mu, x_std)
    Y_tr_n = apply_channel_norm(Y_tr, y_mu, y_std)
    X_te_n = apply_channel_norm(X_te, x_mu, x_std)
    Y_te_n = apply_channel_norm(Y_te, y_mu, y_std)

    # 转 tensor
    X_tr_t = torch.from_numpy(X_tr_n).float()
    Y_tr_t = torch.from_numpy(Y_tr_n).float()
    X_te_t = torch.from_numpy(X_te_n).float()
    Y_te_t = torch.from_numpy(Y_te_n).float()

    # ========== 3) 划分训练/验证集 ==========
    val_ratio = 0.15
    N = X_tr_t.shape[0]
    N_val = int(N * val_ratio)
    N_train = N - N_val
    train_ds, val_ds = random_split(TensorDataset(X_tr_t, Y_tr_t), [N_train, N_val], generator=torch.Generator().manual_seed(42))

    bs = 32
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(X_te_t, Y_te_t), batch_size=bs, shuffle=False, drop_last=False)

    # ========== 4) 定义模型与优化器 ==========
    model = RVAESeq2Seq(x_dim=4, y_dim=1, h_dim=128, z_dim=16, bidirectional=True).to(device)
    optim_ = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ========== 5) 训练：带验证集和早停 ==========
    epochs = 300
    warmup = 10
    patience = 20
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    save_path = "best_model.pt"

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

        # 验证集评估
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

        print(f"Epoch {ep+1:03d} | Train Loss {total/len(train_loader):.4f} | Rec {total_rec/len(train_loader):.4f} | KL {total_kl/len(train_loader):.4f} | "
              f"Val Loss {val_loss:.4f} | Val Rec {val_rec:.4f} | Val KL {val_kl:.4f} | beta={beta:.2f}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stop] 已达到早停条件，最佳epoch: {best_epoch+1}, 最佳验证损失: {best_val_loss:.4f}")
                break

    # ========== 6) 加载最佳模型，测试集评估 ==========
    model.load_state_dict(torch.load(save_path))
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
    print(f"[TEST: original scale] MSE={mse:.6f} | RMSE={rmse:.6f} | MAE={mae:.6f} | PCC={pcc:.6f} | p-value={p_value:.6f}")

    # ========== 7) 画图与保存 ==========
    result_data_dir = "result_data_with_val_1018"
    os.makedirs(result_data_dir, exist_ok=True)

    all_y_pred = []
    all_y_true = []
    with torch.no_grad():
        for xb, yb_n in test_loader:
            xb = xb.to(device)
            yb_n = yb_n.to(device)
            y_pred_n = predict_mean(model, xb).cpu().numpy()
            y_true_n = yb_n.cpu().numpy()
            all_y_pred.append(y_pred_n)
            all_y_true.append(y_true_n)
    y_pred = np.concatenate(all_y_pred, axis=0)
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = inverse_channel_norm(y_pred, y_mu, y_std)
    y_true = inverse_channel_norm(y_true, y_mu, y_std)

    # 仅保存作图所用的原始尺度数据：y_true、y_pred
    save_npz_path = os.path.join(result_data_dir, "y_true_y_pred_original_scale.npz")
    np.savez_compressed(save_npz_path,
                        y_true=y_true.astype(np.float32),
                        y_pred=y_pred.astype(np.float32))
    print(f"[SAVE] 原始尺度的 y_true/y_pred 已保存到 {save_npz_path}")

    x1_arr = X_te[..., 0]
    x2_arr = X_te[..., 1]
    x3_arr = X_te[..., 2]
    x4_arr = X_te[..., 3]
    y_arr  = y_true[..., 0]
    yhat_arr = y_pred[..., 0]

    N_test, T = y_arr.shape
    for i in range(N_test):
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        t = np.arange(T)
        # 让每个子图成为正方形（1:1）
        for ax in axes:
            try:
                ax.set_box_aspect(1)  # Matplotlib >= 3.3
            except AttributeError:
                ax.set_aspect('equal', adjustable='box')  # 旧版本兼容

        axes[0].plot(t, x1_arr[i], linewidth=1.2)
        axes[0].set_ylabel("x1")
        axes[0].set_title(f"Test Sample {i} — Inputs & Residual")

        axes[1].plot(t, x2_arr[i], linewidth=1.2)
        axes[1].set_ylabel("x2")

        axes[2].plot(t, x3_arr[i], linewidth=1.2)
        axes[2].set_ylabel("x3")

        axes[3].plot(t, x4_arr[i], linewidth=1.2)
        axes[3].set_ylabel("x4")

        axes[4].plot(t, y_true[i, :, 0], label="True (denorm)")
        axes[4].plot(t, y_pred[i, :, 0], label="Pred (denorm)")
        axes[4].legend()
        axes[4].set_title(f"Test Sample {i} (original scale)")
        axes[4].set_xlabel("Time")
        axes[4].set_ylabel("Target")

        big_path = os.path.join(result_data_dir, f"test_sample_{i}_fiveplots.png")
        fig.savefig(big_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[PLOT] 已为 {N_test} 个测试样本生成5子图大图至 {result_data_dir}/")