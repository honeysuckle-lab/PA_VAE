# -*- coding: utf-8 -*-
"""
RHC 5通道数据：以 ECG_R 为中心截取 3s 片段并过滤（每个非ECG通道窗口内 R 点≥2）。

目录结构（请按需修改 SEG_DIR）：
E:\PAMprediction\RHCdata\processed_data\mats_5ch\segments\            # 118 *.mat，有 seg_seg(N,5) 或类似
E:\PAMprediction\RHCdata\processed_data\mats_5ch\segments\annotation  # 118 *.mat，有 ECG_R/SCG_lat_R/SCG_hf_R/SCG_dv_R/PA_R

输出：
segments\clips3s\*.mat     # 每个保留片段一个文件
segments\clips3s\index.csv # 片段索引
"""

import os
import re
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.io import loadmat, savemat
import h5py

# -------------------- 配置 --------------------
FS = 500                     # 采样率 Hz
PRE_S = 0.5                  # ECG_R 前 0.5s
POST_S = 2.5                 # ECG_R 后 2.5s
WIN_PRE = int(PRE_S * FS)    # 250
WIN_POST = int(POST_S * FS)  # 1250
WIN_LEN = WIN_PRE + WIN_POST # 1500

SEG_DIR = r"E:\PAMprediction\RHCdata\processed_data\mats_5ch\segments_filtered"
ANN_DIR = os.path.join(SEG_DIR, "annotation")
OUT_DIR = os.path.join(SEG_DIR, "clips3s")
os.makedirs(OUT_DIR, exist_ok=True)

# 如果某些 .mat 的段矩阵名不一致，可在此补充候选名（按优先级）
SEG_VAR_CANDIDATES = ["seg_seg", "sig_seg", "seg", "segment", "data", "sig", "sig_all"]

# -------------------- 读 .mat 工具 --------------------
def is_hdf5(path: str) -> bool:
    """检测 .mat 是否为 v7.3(HDF5)。"""
    with open(path, "rb") as f:
        sig = f.read(8)
    return sig == b"\x89HDF\r\n\x1a\n"

def _transpose_if_needed(arr: np.ndarray) -> np.ndarray:
    """统一通道维到最后：将 (5,N) 转为 (N,5)。"""
    if arr.ndim == 2 and arr.shape[1] != 5 and arr.shape[0] == 5:
        arr = arr.T
    return arr

def read_mat_any(path: str, var_candidates) -> np.ndarray:
    """
    通用读取器：
    - 非 HDF5：用 scipy.io.loadmat；先按候选名取；若没命中，兜底找第一个二维数值矩阵；
    - HDF5：用 h5py；先按候选名取；若没命中，兜底找形如 (N,5) 或 (5,N) 的二维数值矩阵。
    """
    if not is_hdf5(path):
        md = loadmat(path, squeeze_me=True, struct_as_record=False)
        # 优先按候选名
        for v in var_candidates:
            if v in md:
                arr = np.asarray(md[v])
                return _transpose_if_needed(np.squeeze(arr))
        # 兜底：顶层第一个二维数值矩阵
        for k, v in md.items():
            if k.startswith("__"):
                continue
            arr = np.asarray(v)
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return _transpose_if_needed(np.squeeze(arr))
        present = [k for k in md.keys() if not k.startswith("__")]
        raise KeyError(f"{os.path.basename(path)}: none of {var_candidates} found; vars={sorted(present)}")
    else:
        with h5py.File(path, "r") as f:
            for v in var_candidates:
                if v in f:
                    arr = np.array(f[v])
                    arr = np.squeeze(arr)
                    arr = _transpose_if_needed(arr)
                    return arr
            # 兜底：找 (N,5)/(5,N)
            for k in f.keys():
                arr = np.array(f[k])
                if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                    arr = np.squeeze(arr)
                    if arr.shape[1] == 5 or arr.shape[0] == 5:
                        arr = _transpose_if_needed(arr)
                        return arr
            raise KeyError(f"{os.path.basename(path)}: none of {var_candidates} in HDF5 keys={list(f.keys())}")

def read_annotations(path: str) -> Dict[str, np.ndarray]:
    """读取标注 R 点数组；自动兜底/修正形状。"""
    names = ["ECG_R", "SCG_lat_R", "SCG_hf_R", "SCG_dv_R", "PA_R"]
    out: Dict[str, np.ndarray] = {}
    for n in names:
        # 有时变量在结构体里，read_mat_any 的兜底会找到
        arr = read_mat_any(path, [n, f"data.{n}", f"Channel_{n}", n.lower()])
        arr = np.asarray(arr).ravel()
        # 转为整数索引
        if arr.size and not np.issubdtype(arr.dtype, np.integer):
            arr = np.round(arr).astype(int)
        else:
            arr = arr.astype(int)
        out[n] = arr
    return out

# -------------------- 业务逻辑 --------------------
def parse_source_from_filename(fname: str) -> str:
    """
    从文件名抽取来源，例如：
    'TRM107-RHC1_seg01_0340-0425.mat' -> 'TRM107-RHC1'
    """
    base = Path(fname).stem
    m = re.match(r"(.+?)_seg", base)
    return m.group(1) if m else base

def normalize_indices(R: np.ndarray, seg_len: int) -> np.ndarray:
    """
    将 MATLAB 1 基下标转换为 Python 0 基（自动判断）：
    若 max(R) >= seg_len，则认为是 1 基，整体减 1。
    """
    if R.size == 0:
        return R
    R = R.astype(int)
    if np.max(R) >= seg_len:
        R = R - 1
    # 过滤越界
    R = R[(R >= 0) & (R < seg_len)]
    return np.sort(R)

def extract_window(seg: np.ndarray, r_idx: int) -> Tuple[np.ndarray, int, int]:
    """返回 (clip(1500,5), start, end)；越界则返回 (None,None,None)。"""
    start = r_idx - WIN_PRE
    end = r_idx + WIN_POST  # [start, end)
    if start < 0 or end > seg.shape[0]:
        return None, None, None
    return seg[start:end, :], start, end

def count_R_in_window(R: np.ndarray, start: int, end: int) -> int:
    """统计 [start, end) 区间内的 R 点个数。"""
    if R.size == 0:
        return 0
    return int(np.sum((R >= start) & (R < end)))

def process_one_pair(seg_path: str, ann_path: str, writer: csv.DictWriter) -> int:
    """处理一个 segment/annotation 同名文件对；返回保存的片段数量。"""
    # 读段信号并确保形状为 (N,5)
    seg = read_mat_any(seg_path, SEG_VAR_CANDIDATES)
    if seg.ndim != 2:
        raise ValueError(f"{seg_path}: expected 2D array, got {seg.shape}")
    seg = _transpose_if_needed(seg)
    if seg.shape[1] != 5:
        raise ValueError(f"{seg_path}: expected shape (N,5) or (5,N), got {seg.shape}")
    seg = np.asarray(seg, dtype=np.float32)
    N = seg.shape[0]
    if N < WIN_LEN:
        # 过短无法截 3s，直接跳过
        return 0

    # 读标注
    ann = read_annotations(ann_path)
    # 各通道 R 下标归一到 0 基并去越界
    for k in ann.keys():
        ann[k] = normalize_indices(ann[k], N)

    ECG_R = ann["ECG_R"]
    if ECG_R.size == 0:
        return 0

    src = parse_source_from_filename(os.path.basename(seg_path))
    base = Path(seg_path).stem

    saved = 0
    for r in ECG_R:
        clip, s, e = extract_window(seg, int(r))
        if clip is None:
            continue

        cnt_lat = count_R_in_window(ann["SCG_lat_R"], s, e)
        cnt_hf  = count_R_in_window(ann["SCG_hf_R"],  s, e)
        cnt_dv  = count_R_in_window(ann["SCG_dv_R"],  s, e)
        cnt_pa  = count_R_in_window(ann["PA_R"],      s, e)

        # 过滤条件：四个非 ECG 通道均 ≥ 2
        if not (cnt_lat >= 2 and cnt_hf >= 2 and cnt_dv >= 2 and cnt_pa >= 2):
            continue

        # 保存片段
        clip_fname = f"{base}__ECGR{int(r):07d}_{int(s):07d}-{int(e):07d}.mat"
        clip_path = os.path.join(OUT_DIR, clip_fname)
        savemat(clip_path, {
            "clip": clip.astype(np.float32),   # (1500,5)
            "fs": FS,
            "channels": np.array(["ECG", "SCG_lat", "SCG_hf", "SCG_dv", "PA"], dtype=object),
            "source": src,
            "parent_file": os.path.basename(seg_path),
            "ecg_r_index": int(r),
            "win_start": int(s),
            "win_end": int(e),
            "counts": np.array([cnt_lat, cnt_hf, cnt_dv, cnt_pa], dtype=np.int32)
        }, do_compression=True)

        # 写 CSV 索引
        writer.writerow({
            "segment_file": os.path.basename(seg_path),
            "annotation_file": os.path.basename(ann_path),
            "clip_file": os.path.basename(clip_path),
            "source": src,
            "fs": FS,
            "ecg_r_index": int(r),
            "win_start": int(s),
            "win_end": int(e),
            "cnt_SCG_lat_R": cnt_lat,
            "cnt_SCG_hf_R":  cnt_hf,
            "cnt_SCG_dv_R":  cnt_dv,
            "cnt_PA_R":      cnt_pa
        })
        saved += 1

    return saved

# -------------------- 主流程 --------------------
def main():
    seg_files = sorted([f for f in os.listdir(SEG_DIR) if f.lower().endswith("_filt.mat")])
    ann_names = {f for f in os.listdir(ANN_DIR) if f.lower().endswith(".mat")}

    csv_path = os.path.join(OUT_DIR, "index.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        fieldnames = [
            "segment_file","annotation_file","clip_file","source","fs",
            "ecg_r_index","win_start","win_end",
            "cnt_SCG_lat_R","cnt_SCG_hf_R","cnt_SCG_dv_R","cnt_PA_R"
        ]
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        total_pairs, total_saved = 0, 0
        for seg_name in seg_files:
            # 去掉 _filt 后缀，匹配 annotation
            ann_name = seg_name.replace("_filt.mat", ".mat")
            ann_path = os.path.join(ANN_DIR, ann_name)
            seg_path = os.path.join(SEG_DIR, seg_name)
            if not os.path.exists(ann_path):
                print(f"[WARN] No matching annotation for {seg_name}")
                continue
            total_pairs += 1
            try:
                saved = process_one_pair(seg_path, ann_path, writer)
                total_saved += saved
                print(f"[OK] {seg_name}: saved {saved} clips")
            except Exception as e:
                print(f"[ERROR] {seg_name}: {e}")

    print("\n========== SUMMARY ==========")
    print(f"Pairs processed : {total_pairs}")
    print(f"Clips saved     : {total_saved}")
    print(f"CSV index       : {csv_path}")
    print(f"Clips directory : {OUT_DIR}")

if __name__ == "__main__":
    main()
