import os, glob
import numpy as np
import scipy.signal as sig
from scipy.io import loadmat, savemat
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import pandas as pd

# 配置
DATA_DIR = r"E:\PAMprediction\RHCdata\processed_data\mats_5ch\segments"
OUT_DIR  = r"E:\PAMprediction\RHCdata\processed_data\mats_5ch\segments_filtered"
NOTCHS  = [50.0, 100.0]
NOTCH_Q = 35.0
ECG_BP  = (0.5, 100.0)
SCG_HP  = 0.7
SCG_BP  = (1.0, 30.0)
ORDER_BP = 4

NFFT = 1024
NPERSEG = 512
NOVERLAP = 256

DO_PLOTS        = True
PLOT_EVERY      = 1
DO_ATTENUATION  = True
ATTEN_EVERY     = 10
DO_BODE         = True

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def design_notch_sos(fs, f0, Q):
    w0 = f0 / (fs / 2.0)
    b, a = sig.iirnotch(w0, Q)
    return sig.tf2sos(b, a)

def design_butter_bandpass_sos(fs, f_lo, f_hi, order=4):
    wn = [f_lo/(fs/2.0), f_hi/(fs/2.0)]
    return sig.butter(order, wn, btype="bandpass", output="sos")

def design_butter_highpass_sos(fs, f_lo, order=4):
    wn = f_lo/(fs/2.0)
    return sig.butter(order, wn, btype="highpass", output="sos")

def apply_zero_phase_sos(x, sos_list):
    y = x
    for sos in sos_list:
        y = sig.sosfiltfilt(sos, y, padtype="odd", padlen=None)
    return y

def welch_psd(x, fs, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT):
    f, Pxx = sig.welch(
        x, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        detrend="constant"
    )
    return f, Pxx

def spectral_centroid(f, Pxx):
    denom = np.sum(Pxx) + 1e-12
    return float(np.sum(f * Pxx) / denom)

def band_energy_ratio(f, Pxx, band_lo, band_hi):
    mask = (f >= band_lo) & (f < band_hi)
    num = np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0
    den = np.trapz(Pxx, f) + 1e-12
    return float(num / den)

def bode_plot_once(out_dir, fs):
    """一次性输出 ECG/SCG 滤波链的 Bode 图."""
    ensure_dir(out_dir)
    ecg_chain = [design_notch_sos(fs, f0, NOTCH_Q) for f0 in NOTCHS] + \
                [design_butter_bandpass_sos(fs, ECG_BP[0], ECG_BP[1], ORDER_BP)]
    scg_chain = [design_notch_sos(fs, f0, NOTCH_Q) for f0 in NOTCHS] + \
                [design_butter_highpass_sos(fs, SCG_HP, ORDER_BP),
                 design_butter_bandpass_sos(fs, SCG_BP[0], SCG_BP[1], ORDER_BP)]

    def chain_freqz(sos_list, worN=8192, fs=fs):
        w = np.linspace(0, np.pi, worN)
        freqs = w * fs / (2*np.pi)
        H = np.ones_like(w, dtype=np.complex128)
        for sos in sos_list:
            _, h = sig.sosfreqz(sos, worN=w)
            H *= h
        return freqs, H

    fe, He = chain_freqz(ecg_chain, fs=fs)
    fs_, Hs = chain_freqz(scg_chain, fs=fs)

    plots = [
        ("ECG Chain Magnitude Response", fe, He, "bode_ecg_magnitude.png", "mag"),
        ("ECG Chain Phase Response",     fe, He, "bode_ecg_phase.png",     "phase"),
        ("SCG Chain Magnitude Response", fs_, Hs, "bode_scg_magnitude.png","mag"),
        ("SCG Chain Phase Response",     fs_, Hs, "bode_scg_phase.png",    "phase"),
    ]
    for title, fr, H, stem, mode in plots:
        fig = plt.figure(figsize=(8,5))
        try:
            plt.title(title)
            if mode == "mag":
                plt.plot(fr, 20*np.log10(np.maximum(np.abs(H), 1e-12)))
                plt.ylabel("Magnitude (dB)")
            else:
                plt.plot(fr, np.unwrap(np.angle(H))*180/np.pi)
                plt.ylabel("Phase (deg)")
            plt.xlabel("Frequency (Hz)")
            plt.grid(True, which="both")
            plt.savefig(os.path.join(out_dir, stem), dpi=180, bbox_inches="tight")
        finally:
            plt.close(fig)

def process_one(mat_path, out_dir, plot_every=PLOT_EVERY, atten_every=ATTEN_EVERY):
    d = loadmat(mat_path)
    if "sig_seg" not in d or "fs" not in d:
        raise KeyError("sig_seg or fs not found in mat file")
    sig_seg = np.asarray(d["sig_seg"], dtype=np.float64)
    fs = int(np.ravel(d["fs"])[0])
    N, C = sig_seg.shape
    if C != 5:
        raise ValueError(f"sig_seg shape expected (*,5), got {sig_seg.shape}")

    # ECG滤波链（含notch）
    ecg_chain = [design_notch_sos(fs, f0, NOTCH_Q) for f0 in NOTCHS] + \
                [design_butter_bandpass_sos(fs, ECG_BP[0], ECG_BP[1], ORDER_BP)]
    # SCG滤波链（不含notch）
    scg_chain = [
        design_butter_highpass_sos(fs, SCG_HP, ORDER_BP),
        design_butter_bandpass_sos(fs, SCG_BP[0], SCG_BP[1], ORDER_BP)
    ]

    # 频谱（前）——对 ECG + 三轴全部计算（绘图/衰减用）
    pre_psd = {}
    for ch, idx in {"ECG":0, "SCG_lat":1, "SCG_hf":2, "SCG_dv":3}.items():
        pre_psd[ch] = welch_psd(sig_seg[:, idx], fs)

    # 滤波：前 4 通道，PA(4) 保持
    out = sig_seg.copy()
    out[:, 0] = apply_zero_phase_sos(sig_seg[:, 0], ecg_chain)     # ECG
    for ax in [1, 2, 3]:                                           # SCG axes
        out[:, ax] = apply_zero_phase_sos(sig_seg[:, ax], scg_chain)
    # PA(4) 不变

    # 频谱（后）——ECG + 三轴全部
    post_psd = {}
    for ch, idx in {"ECG":0, "SCG_lat":1, "SCG_hf":2, "SCG_dv":3}.items():
        post_psd[ch] = welch_psd(out[:, idx], fs)

    # 写出 _filt.mat（sig_seg_filt）
    base = os.path.splitext(os.path.basename(mat_path))[0]
    ensure_dir(out_dir)
    savemat(
        os.path.join(out_dir, base + "_filt.mat"),
        {
            "sig_seg_filt": out.astype(np.float32),
            "fs": fs,
            "channels": np.array(["ECG", "SCG_lat", "SCG_hf", "SCG_dv", "PA"], dtype=object),
        },
        do_compression=True
    )

    # ---- 指标（后）——ECG + 三轴 ----
    def metrics_of(x, f, Pxx):
        return dict(
            rms=float(np.sqrt(np.mean(x**2))),
            robust_p2p=float(np.percentile(x, 95) - np.percentile(x, 5)),
            skew=float(skew(x)),
            kurtosis=float(kurtosis(x, fisher=False)),
            spectral_centroid=spectral_centroid(f, Pxx),
            E_0p5_12_ratio=band_energy_ratio(f, Pxx, 0.5, 12.0),
            E_12_30_ratio=band_energy_ratio(f, Pxx, 12.0, 30.0),
        )

    metrics = {"file": base + ".mat"}
    for ch, idx in {"ECG":0, "SCG_lat":1, "SCG_hf":2, "SCG_dv":3}.items():
        f_post, P_post = post_psd[ch]
        md = metrics_of(out[:, idx], f_post, P_post)
        metrics.update({f"{ch}_{k}": v for k, v in md.items()})

    # ---- 频谱对比图（可选）----
    if DO_PLOTS and (hash(base) % max(1, plot_every) == 0):
        chans_to_plot = ["ECG", "SCG_lat", "SCG_hf", "SCG_dv"]
        for ch in chans_to_plot:
            fig = plt.figure(figsize=(8,5))
            try:
                f0, P0 = pre_psd[ch]
                f1, P1 = post_psd[ch]
                plt.title(f"{ch} Spectrum: Pre vs Post")
                plt.semilogy(f0, P0); plt.semilogy(f1, P1)
                plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD"); plt.grid(True, which="both")
                plt.legend(["Pre","Post"])
                plt.savefig(os.path.join(out_dir, base + f"_{ch}_spectrum.png"),
                            dpi=160, bbox_inches="tight")
            finally:
                plt.close(fig)

    # ---- 有效衰减曲线（Post/Pre 的 dB 比值）----
    if DO_ATTENUATION and (hash(base) % max(1, atten_every) == 0):
        EPS = 1e-18
        chans_to_plot = ["ECG", "SCG_lat", "SCG_hf", "SCG_dv"]
        for ch in chans_to_plot:
            fig = plt.figure(figsize=(8,5))
            try:
                f_pre, P_pre = pre_psd[ch]
                f_post, P_post = post_psd[ch]
                att_db = 10.0 * np.log10((P_post + EPS) / (P_pre + EPS))
                plt.title(f"{ch} Effective Attenuation (Post/Pre, dB)")
                plt.plot(f_pre, att_db)
                plt.xlabel("Frequency (Hz)"); plt.ylabel("Attenuation (dB)")
                plt.grid(True, which="both")
                plt.savefig(os.path.join(out_dir, base + f"_{ch}_attenuation.png"),
                            dpi=160, bbox_inches="tight")
            finally:
                plt.close(fig)

    return metrics

def bode_plot_once_with_zero_phase(out_dir, fs):
    """
    对 ECG/SCG 滤波链画 Bode：
    - Magnitude：单次 |H|（20log10） & 零相位等效 |H|^2（40log10）
    - Phase：单次相位 & 零相位等效（0° 基线）
    """
    ensure_dir(out_dir)
    # ECG链（含notch）
    ecg_chain = [design_notch_sos(fs, f0, NOTCH_Q) for f0 in NOTCHS] + \
                [design_butter_bandpass_sos(fs, ECG_BP[0], ECG_BP[1], ORDER_BP)]
    # SCG链（不含notch）
    scg_chain = [
        design_butter_highpass_sos(fs, SCG_HP, ORDER_BP),
        design_butter_bandpass_sos(fs, SCG_BP[0], SCG_BP[1], ORDER_BP)
    ]

    def chain_freqz(sos_list, worN=8192, fs=fs):
        w = np.linspace(0, np.pi, worN)
        f = w * fs / (2*np.pi)
        H = np.ones_like(w, dtype=np.complex128)
        for sos in sos_list:
            _, h = sig.sosfreqz(sos, worN=w)
            H *= h
        return f, H

    fE, HE = chain_freqz(ecg_chain, fs=fs)
    fS, HS = chain_freqz(scg_chain, fs=fs)

    def plot_pair(f, H, stem):
        mag_single = 20*np.log10(np.maximum(np.abs(H), 1e-12))
        mag_eff    = 40*np.log10(np.maximum(np.abs(H), 1e-12))   # filtfilt 等效
        ph_single  = np.unwrap(np.angle(H))*180/np.pi

        # Magnitude
        fig = plt.figure(figsize=(8,5))
        try:
            plt.title(f"{stem} Magnitude")
            plt.plot(f, mag_single, label="single-pass |H| (dB)")
            plt.plot(f, mag_eff,   label="zero-phase equiv |H|^2 (dB)", linestyle="--")
            plt.xlabel("Frequency (Hz)"); plt.ylabel("dB"); plt.grid(True, which="both")
            plt.legend()
            plt.savefig(os.path.join(out_dir, stem.lower()+"_mag.png"), dpi=180, bbox_inches="tight")
        finally:
            plt.close(fig)

        # Phase
        fig = plt.figure(figsize=(8,5))
        try:
            plt.title(f"{stem} Phase")
            plt.plot(f, ph_single, label="single-pass phase")
            plt.plot(f, np.zeros_like(f), label="zero-phase equiv (filtfilt)", linestyle="--")
            plt.xlabel("Frequency (Hz)"); plt.ylabel("deg"); plt.grid(True, which="both")
            plt.legend()
            plt.savefig(os.path.join(out_dir, stem.lower()+"_phase.png"), dpi=180, bbox_inches="tight")
        finally:
            plt.close(fig)

    plot_pair(fE, HE, "ECG Chain")
    plot_pair(fS, HS, "SCG Chain")

def batch_process(data_dir=DATA_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))

    # 一次性 Bode 图（参考 ds125 风格）
    if DO_BODE and files:
        first_file = files[0]
        d = loadmat(first_file)
        fs = int(np.ravel(d["fs"])[0]) if "fs" in d else 500
        bode_plot_once_with_zero_phase(os.path.join(out_dir, "bode"), fs)

    all_metrics = []
    for i, fp in enumerate(files, 1):
        try:
            m = process_one(fp, out_dir, plot_every=PLOT_EVERY, atten_every=ATTEN_EVERY)
            all_metrics.append(m)
        except Exception as e:
            print(f"[WARN] {fp}: {e}")
        if i % 100 == 0:
            print(f"[INFO] processed {i}/{len(files)}")

    # ---- 写出 CSV 与分位数（ECG+三轴）----
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

        out_rows = []
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        channels_for_q = ["ECG", "SCG_lat", "SCG_hf", "SCG_dv"]
        metrics_list   = ["rms","robust_p2p","skew","kurtosis","spectral_centroid","E_0p5_12_ratio","E_12_30_ratio"]

        for ch in channels_for_q:
            for k in metrics_list:
                col = f"{ch}_{k}"
                if col in df.columns:
                    qs = df[col].quantile(quantiles).to_dict()
                    row = {"channel": ch, "metric": k}
                    row.update({f"q{int(100*q)}": float(val) for q, val in qs.items()})
                    out_rows.append(row)

        pd.DataFrame(out_rows).to_csv(os.path.join(out_dir, "metrics_quantiles.csv"), index=False)

if __name__ == "__main__":
    batch_process(DATA_DIR, OUT_DIR)