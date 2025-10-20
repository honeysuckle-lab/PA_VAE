import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import scipy.signal as sig

INDEX_CSV = r"E:\PAMprediction\RHCdata\processed_data\mats_5ch\segments_filtered\clips3s\index.csv"
CLIP_DIR = r"E:\PAMprediction\RHCdata\processed_data\mats_5ch\segments_filtered\clips3s"
OUT_SUMMARY = "metrics_summary.csv"
OUT_QUANTILES = "metrics_quantiles.csv"
OUT_QUALITY = "quality_labels.csv"

CHANNELS = ["ECG", "SCG_lat", "SCG_hf", "SCG_dv", "PA"]
METRICS = ["rms","robust_p2p","skew","kurtosis","spectral_centroid","E_0p5_12_ratio","E_12_30_ratio"]

def welch_psd(x, fs, nperseg=512, noverlap=256, nfft=1024):
    f, Pxx = sig.welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend="constant")
    return f, Pxx

def spectral_centroid(f, Pxx):
    denom = np.sum(Pxx) + 1e-12
    return float(np.sum(f * Pxx) / denom)

def band_energy_ratio(f, Pxx, band_lo, band_hi):
    mask = (f >= band_lo) & (f < band_hi)
    num = np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0
    den = np.trapz(Pxx, f) + 1e-12
    return float(num / den)

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

def process_clip_file(clip_path):
    d = loadmat(clip_path)
    clip = np.asarray(d["clip"], dtype=np.float32)
    fs = int(np.ravel(d["fs"])[0]) if "fs" in d else 500
    out = {}
    # ECG+SCG三轴
    for ch, idx in zip(CHANNELS[:4], range(4)):
        x = clip[:, idx]
        f, Pxx = welch_psd(x, fs)
        out.update({f"{ch}_{k}": v for k, v in metrics_of(x, f, Pxx).items()})
    # PA
    pa = clip[:, 4]
    out["PA_max"] = float(np.max(pa))
    out["PA_min"] = float(np.min(pa))
    return out

def main():
    df_index = pd.read_csv(INDEX_CSV)
    metrics_rows = []
    for i, row in df_index.iterrows():
        clip_file = row["clip_file"]
        clip_path = os.path.join(CLIP_DIR, clip_file)
        if not os.path.exists(clip_path):
            print(f"[WARN] {clip_path} not found")
            continue
        metrics = process_clip_file(clip_path)
        metrics["file"] = clip_file
        metrics_rows.append(metrics)
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(os.path.join(CLIP_DIR, OUT_SUMMARY), index=False)

    # 四通道分位数
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    out_rows = []
    for ch in CHANNELS[:4]:
        for k in METRICS:
            col = f"{ch}_{k}"
            if col in df_metrics.columns:
                qs = df_metrics[col].quantile(quantiles).to_dict()
                row = {"channel": ch, "metric": k}
                row.update({f"q{int(100*q)}": float(val) for q, val in qs.items()})
                out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(os.path.join(CLIP_DIR, OUT_QUANTILES), index=False)

    # 结合 sqi_ranker.py 算法评分
    # 载入分位数
    qdf = pd.read_csv(os.path.join(CLIP_DIR, OUT_QUANTILES))
    # 构建分位数映射
    qmap = {ch:{} for ch in CHANNELS[:4]}
    for _,r in qdf.iterrows():
        ch = r["channel"]; met = r["metric"]
        qmap[ch][met] = {f"q{int(100*q)}": r[f"q{int(100*q)}"] for q in quantiles}
        qmap[ch][met]["q50"] = r["q50"]
        qmap[ch][met]["q25"] = r["q25"]
        qmap[ch][met]["q75"] = r["q75"]
        qmap[ch][met]["q05"] = r["q5"] if "q5" in r else r["q05"]
        qmap[ch][met]["q95"] = r["q95"]

    # SQI评分参数（可直接引用 sqi_ranker.py 的 PREFS、WEIGHTS、label_from_score 等）
    PREFS = {
        "ECG": {
            "spectral_centroid": "inrange",
            "E_0p5_12_ratio":    "high",
            "E_12_30_ratio":     "low",
            "kurtosis":          "low",
            "rms":               "inrange",
            "robust_p2p":        "inrange",
            "skew":              "inrange",
        },
        "SCG_lat": {
            "spectral_centroid": "inrange",
            "E_0p5_12_ratio":    "inrange",
            "E_12_30_ratio":     "inrange",
            "kurtosis":          "low",
            "rms":               "inrange",
            "robust_p2p":        "inrange",
            "skew":              "inrange",
        },
        "SCG_hf": {
            "spectral_centroid": "inrange",
            "E_0p5_12_ratio":    "inrange",
            "E_12_30_ratio":     "inrange",
            "kurtosis":          "low",
            "rms":               "inrange",
            "robust_p2p":        "inrange",
            "skew":              "inrange",
        },
        "SCG_dv": {
            "spectral_centroid": "inrange",
            "E_0p5_12_ratio":    "inrange",
            "E_12_30_ratio":     "inrange",
            "kurtosis":          "low",
            "rms":               "inrange",
            "robust_p2p":        "inrange",
            "skew":              "inrange",
        },
    }
    WEIGHTS = {
        "ECG":     {"spectral_centroid":0.30, "E_0p5_12_ratio":0.25, "E_12_30_ratio":0.20, "kurtosis":0.10, "rms":0.10, "robust_p2p":0.05},
        "SCG_lat": {"spectral_centroid":0.25, "E_0p5_12_ratio":0.20, "E_12_30_ratio":0.20, "kurtosis":0.10, "rms":0.15, "robust_p2p":0.10},
        "SCG_hf":  {"spectral_centroid":0.25, "E_0p5_12_ratio":0.20, "E_12_30_ratio":0.20, "kurtosis":0.10, "rms":0.15, "robust_p2p":0.10},
        "SCG_dv":  {"spectral_centroid":0.25, "E_0p5_12_ratio":0.20, "E_12_30_ratio":0.20, "kurtosis":0.10, "rms":0.15, "robust_p2p":0.10},
    }
    GOOD_THRESH = 0.75
    OK_THRESH   = 0.50

    def soft_score_inrange(x, a, b, lo, hi):
        if np.isnan(x) or any(np.isnan(v) for v in [a,b,lo,hi]):
            return 0.0
        if a > b:
            a, b = b, a
        if lo > hi:
            lo, hi = hi, lo
        if x >= a and x <= b:
            return 1.0
        if x < a and x >= lo:
            return 0.5 + 0.5 * (x - lo) / (a - lo + 1e-12)
        if x > b and x <= hi:
            return 0.5 + 0.5 * (hi - x) / (hi - b + 1e-12)
        iqr = (b - a) + 1e-12
        if x < lo:
            return max(0.0, 0.5 * np.exp((x - lo) / (0.5*iqr)))
        else:
            return max(0.0, 0.5 * np.exp((hi - x) / (0.5*iqr)))

    def soft_score_high(x, target, lo, a):
        if np.isnan(x) or any(np.isnan(v) for v in [target, lo, a]):
            return 0.0
        if x >= target:
            return 1.0
        if x >= a:
            return 0.5 + 0.5 * (x - a) / (target - a + 1e-12)
        span = (a - lo) + 1e-12
        return max(0.0, 0.5 * np.exp((x - a) / (0.5*span)))

    def soft_score_low(x, target, b, hi):
        if np.isnan(x) or any(np.isnan(v) for v in [target, b, hi]):
            return 0.0
        if x <= target:
            return 1.0
        if x <= b:
            return 0.5 + 0.5 * (b - x) / (b - target + 1e-12)
        span = (hi - b) + 1e-12
        return max(0.0, 0.5 * np.exp((b - x) / (0.5*span)))

    def score_channel_row(row, ch, qmap, prefs, weights):
        scores = {}
        for met, pref in prefs.items():
            col = f"{ch}_{met}"
            x = row.get(col, np.nan)
            q = qmap.get(ch,{}).get(met, None)
            if q is None:
                s = 0.0
            else:
                a,b,lo,hi,med = q["q25"], q["q75"], q["q05"], q["q95"], q["q50"]
                if pref == "inrange":
                    s = soft_score_inrange(x, a,b, lo,hi)
                elif pref == "high":
                    s = soft_score_high(x, med, lo, a)
                elif pref == "low":
                    s = soft_score_low(x, med, b, hi)
                else:
                    s = soft_score_inrange(x, a,b, lo,hi)
            scores[met] = s
        # weighted average
        ch_score = 0.0
        wsum = 0.0
        for met, w in weights.items():
            if met in scores:
                ch_score += w * scores[met]
                wsum += w
        ch_score = ch_score / (wsum + 1e-12)
        return ch_score

    def label_from_score(overall, good=GOOD_THRESH, ok=OK_THRESH):
        if overall >= good:
            return "Good"
        elif overall >= ok:
            return "OK"
        else:
            return "Bad"

    # 计算评分
    rows = []
    for i, r in df_metrics.iterrows():
        per_ch_scores = {}
        for ch in CHANNELS[:4]:
            s = score_channel_row(r, ch, qmap, PREFS[ch], WEIGHTS[ch])
            per_ch_scores[ch] = s
        scg_best = max(per_ch_scores["SCG_lat"], per_ch_scores["SCG_hf"], per_ch_scores["SCG_dv"])
        overall = 0.5 * per_ch_scores["ECG"] + 0.5 * scg_best
        label = label_from_score(overall, GOOD_THRESH, OK_THRESH)
        # PA最小值小于0直接Bad
        if r.get("PA_min", np.nan) < 0:
            label = "Bad"
        rows.append({
            "file": r["file"],
            "overall_score": round(float(overall), 3),
            "label": label,
            "ECG_score": round(float(per_ch_scores["ECG"]), 3),
            "SCG_lat_score": round(float(per_ch_scores["SCG_lat"]), 3),
            "SCG_hf_score": round(float(per_ch_scores["SCG_hf"]), 3),
            "SCG_dv_score": round(float(per_ch_scores["SCG_dv"]), 3),
            "PA_max": r.get("PA_max", np.nan),
            "PA_min": r.get("PA_min", np.nan),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(CLIP_DIR, OUT_QUALITY), index=False)
    print(f"Wrote: {os.path.join(CLIP_DIR, OUT_QUALITY)}")
    print(out["label"].value_counts())

if __name__ == "__main__":
    main()