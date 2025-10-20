import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

CLIP_DIR = r"E:\PAMprediction\RHCdata\processed_data\mats_5ch\segments_filtered\clips3s"
QUALITY_CSV = os.path.join(CLIP_DIR, "quality_labels.csv")
OUT_DIR = os.path.join(CLIP_DIR, "split_results_good")
os.makedirs(OUT_DIR, exist_ok=True)

df_quality = pd.read_csv(QUALITY_CSV)
df_good = df_quality[df_quality["label"] == "Good"].copy()
sources = df_good["source"].unique()
test_size = 0.2
best_score = None
best_train_sources = None
best_test_sources = None

for trial in range(1000):
    np.random.shuffle(sources)
    n_test = max(1, int(len(sources) * test_size))
    test_sources = sources[:n_test]
    train_sources = sources[n_test:]
    df_train = df_good[df_good["source"].isin(train_sources)]
    df_test = df_good[df_good["source"].isin(test_sources)]
    # 分布覆盖约束
    if df_test["PA_min"].min() < df_train["PA_min"].min(): continue
    if df_test["PA_max"].max() > df_train["PA_max"].max(): continue
    # 形状相似性（用二维直方图的EMD距离衡量）
    H_train, _, _ = np.histogram2d(df_train["PA_min"], df_train["PA_max"], bins=30)
    H_test, _, _ = np.histogram2d(df_test["PA_min"], df_test["PA_max"], bins=30)
    score = wasserstein_distance(H_train.ravel(), H_test.ravel())
    # 比例约束
    ratio = len(df_train) / (len(df_train) + len(df_test))
    if ratio < 0.75 or ratio > 0.85: continue
    if best_score is None or score < best_score:
        best_score = score
        best_train_sources = train_sources.copy()
        best_test_sources = test_sources.copy()

# 用最佳分组
df_good["set"] = df_good["source"].apply(lambda x: "test" if x in best_test_sources else "train")
df_train = df_good[df_good["set"] == "train"]
df_test = df_good[df_good["set"] == "test"]

# 保存csv
df_train.to_csv(os.path.join(OUT_DIR, "train_list.csv"), index=False)
df_test.to_csv(os.path.join(OUT_DIR, "test_list.csv"), index=False)

# 画分布图
plt.figure(figsize=(8,6))
sns.kdeplot(
    x=df_train["PA_min"], y=df_train["PA_max"],
    cmap="Blues", fill=True, alpha=0.5
)
sns.kdeplot(
    x=df_test["PA_min"], y=df_test["PA_max"],
    cmap="Oranges", fill=True, alpha=0.5
)
plt.xlabel("PA_min")
plt.ylabel("PA_max")
plt.title("PA_min vs PA_max 2D Density (Train & Test, Good only)")
from matplotlib.patches import Patch
plt.legend(handles=[
    Patch(facecolor="blue", edgecolor="blue", alpha=0.5, label="Train"),
    Patch(facecolor="orange", edgecolor="orange", alpha=0.5, label="Test")
])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pa_min_max_2d_density.png"), dpi=150)
plt.close()

print(f"[OK] Saved split and plot to {OUT_DIR}")