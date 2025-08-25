# -*- coding: utf-8 -*-
"""
GPT-SoVITS 微调训练集 质检+筛选（双击直跑版）
特性：
1) 计算每个 wav 的：时长(s)、平均RMS能量、F0(基频)中位数、语速(cps=汉字数/秒)
2) 依据【时长/能量/F0/语速】打标签：keep_train / move_ref / discard，并细分 reason
3) 只生成报告 or 实际搬运：在【用户配置】区切换布尔开关即可
4) 支持交互选目录（可开关），全程相对/绝对路径均可
"""

import os
import re
import sys
import shutil
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import csv  # [ADD] 写 CSV 时统一加引号，防止文本里逗号造成解析问题

# ============【用户配置】（双击运行前改这里）===========
WAV_DIR       = r"D:\lyf\GPT-SoVITS-main\datasets\paimeng"   # 留空则弹窗选择；或填绝对路径，如 r"D:\lyf\GPT-SoVITS-main\datasets\paimeng_new"
CSV_PATH      = r"D:\lyf\GPT-SoVITS-main\datasets\paimeng.csv"   # 训练对照CSV（两列：文件名/语音文件, 文本）。为空则不做语速判断
SR            = 32000 # 与模型config一致（你给的config是 32000）

MIN_DUR       = 2.2   # 最小时长阈值（秒）
MAX_DUR       = 8.0  # 最长时长阈值（秒）
PCTL_HIGH     = 87    # 能量/F0 高分位阈值（百分位，80~90常用）
SPEED_FAST    = 6.5   # 语速上限（>7字/秒视为很快）
SPEED_SLOW    = 2.2   # 语速下限（<2字/秒视为很慢）
KEEP_SHORT_BUT_FAST_TO_TRAIN = False  # “短但很快”的完整小句直接留训练，不进参考库
SHORT_FLOOR = 1.2   # 极短下限：再短直接丢
LONG_CEIL   = 12.0  # 极长上限：再长直接丢

REPORT_CSV    = r"D:\lyf\GPT-SoVITS-main\datasets\paimeng_report.csv"       # 输出报告路径（CSV）
TRAIN_OUT     = r"D:\lyf\GPT-SoVITS-main\datasets\paimeng_train"      # 实际搬运：主训练集输出目录
REF_OUT       = r"D:\lyf\GPT-SoVITS-main\datasets\paimeng_ref"         # 实际搬运：参考音频输出目录
DISCARD_OUT   = r"D:\lyf\GPT-SoVITS-main\datasets\paimeng_discard"   # [ADD] 丢弃样本的音频输出目录

TRAIN_CSV_OUT = r"D:\lyf\GPT-SoVITS-main\datasets\paimeng_train.csv"  # [ADD] 训练集 CSV 输出
REF_CSV_OUT   = r"D:\lyf\GPT-SoVITS-main\datasets\paimeng_ref.csv"     # [ADD] 参考集 CSV 输出
DISCARD_CSV_OUT = r"D:\lyf\GPT-SoVITS-main\datasets\paimeng_discard.csv"   # [MODIFY] 原来是空字符串，改为实际路径以导出丢弃清单

APPLY_MOVE    = True # False=出报告；True=搬运文件
USE_COPY      = True # True=复制，False=移动
PICK_DIR_GUI  = True  # True=启动时弹出文件夹选择器（当 WAV_DIR 为空或你想手选时）

# ============【内部参数】（一般不用改）===========
FRAME_LENGTH = 2048
HOP_LENGTH   = 512

# [ADD] 静音判定阈值
SIL_FRAC_MAX = 0.30        # 整段静音比例上限（0~1）
LEAD_TAIL_SIL_MAX = 0.30   # 句首/句尾静音上限（秒）
# [ADD] 低能量阈值（RMS），严格可调小/大
SIL_RMS_THR = 0.01         # 经验阈值；若你能量值范围不同，可调 0.005~0.02
# [ADD] 削波判定阈值
CLIP_PEAK_THR = 0.99          # 峰值判定阈值
CLIP_NEAR_THR = 0.985         # 近峰值阈值
CLIP_NEAR_FRAC_THR = 0.001    # 近峰值样本占比阈值（>0.1% 视为削波）

# ============ 工具函数 ============
def pick_dir_dialog() -> str:
    """Windows/macOS/Linux 弹窗选目录（仅在 tkinter 可用时）"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        folder = filedialog.askdirectory(title="请选择包含 WAV 的文件夹")
        return folder or ""
    except Exception:
        return ""

def load_csv_text_map(csv_path: Path):
    """从CSV读取 文件名->文本 的映射。支持列名：文件名 或 语音文件；文本列名：文本"""
    if not csv_path or not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # 列适配
    file_col = None
    # [MODIFY] 兼容 “新文件名”
    for cand in ["文件名", "语音文件", "新文件名"]:
        if cand in df.columns:
            file_col = cand; break
    if file_col is None or "文本" not in df.columns:
        print("[WARN] CSV 缺少必要列（文件名/语音文件, 文本），将跳过语速计算。")
        return {}
    mp = {}
    for _, r in df.iterrows():
        name = str(r[file_col]).strip()
        base = os.path.basename(name)
        mp[base] = str(r["文本"])
    return mp

def count_cn_chars(text: str) -> int:
    """统计用于语速的“可发音汉字”数量（去标点/空格/控制符；英文数字不计入语速）"""
    if not text: return 0
    t = unicodedata.normalize("NFKC", text)
    # 去掉标点/空白
    t = re.sub(r"\s+", "", t)
    # 仅计中文字符（基本汉字区）
    chars = re.findall(r"[\u4e00-\u9fff]", t)
    return len(chars)

def audio_stats(wav_path: Path, sr_target=SR):
    """返回 (duration_sec, mean_rms, median_f0). 无F0则返回0."""
    y, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
    dur = len(y) / sr
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH).squeeze()
    mean_rms = float(np.mean(rms)) if rms.size else 0.0
    f0 = librosa.yin(y, fmin=50, fmax=600, sr=sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    f0 = f0[np.isfinite(f0)]
    median_f0 = float(np.median(f0)) if f0.size else 0.0
    return dur, mean_rms, median_f0

def rms_per_frame(wav: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    计算逐帧 RMS。输入 wav 为一维 float32/float64（若是多通道，请先转单通道）。
    """
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # 简单转单声道
    n = len(wav)
    if n < frame_length:
        # 不足一帧：补零到一帧
        pad = frame_length - n
        wav = np.pad(wav, (0, pad))
        n = len(wav)
    # 帧起点索引
    starts = np.arange(0, n - frame_length + 1, hop_length, dtype=int)
    # 向量化取帧（比 for 循环快）
    frames = np.stack([wav[s:s + frame_length] for s in starts], axis=0)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)  # 避免 0 下溢
    return rms

def silence_metrics(rms: np.ndarray, sr: int, hop_length: int, thr: float):
    """
    根据逐帧 RMS 计算：
    - sil_frac：整段静音比例
    - lead_sec：句首静音时长（秒）
    - tail_sec：句尾静音时长（秒）
    """
    if rms.size == 0:
        return 1.0, 1e9, 1e9  # 全静音视为极端
    hop_sec = hop_length / float(sr)
    sil = (rms < thr).astype(np.uint8)
    # 整段静音比例
    sil_frac = sil.mean()
    # 句首静音帧数
    lead = 0
    for v in sil:
        if v == 1: lead += 1
        else: break
    # 句尾静音帧数
    tail = 0
    for v in sil[::-1]:
        if v == 1: tail += 1
        else: break
    lead_sec = lead * hop_sec
    tail_sec = tail * hop_sec
    return sil_frac, lead_sec, tail_sec

# [ADD] 削波检测
def detect_clipping(y: np.ndarray,
                    peak_thr: float = CLIP_PEAK_THR,
                    near_thr: float = CLIP_NEAR_THR,
                    near_frac_thr: float = CLIP_NEAR_FRAC_THR):
    """
    返回 (peak, near_frac, is_clipped)
    - peak: 振幅峰值 |y|
    - near_frac: |y| >= near_thr 的样本比例
    - is_clipped: (peak >= peak_thr) 且 (near_frac > near_frac_thr)
    """
    if y is None or y.size == 0:
        return 0.0, 0.0, False
    y = np.asarray(y)
    peak = float(np.max(np.abs(y)))
    near_frac = float((np.abs(y) >= near_thr).mean())
    is_clipped = (peak >= peak_thr) and (near_frac > near_frac_thr)
    return peak, near_frac, is_clipped

def decide_action(item, rms_hi, f0_hi, min_dur, max_dur, speed_fast, speed_slow):
    """
    基于：时长/能量/F0/语速 → action, reason
    """
    dur = item["duration"]
    rms = item["mean_rms"]
    f0  = item["median_f0"]
    cps = item.get("cps", None)
    # ---- [ADD] 信号质量硬规则：削波 ----
    if item.get("is_clipped", False):
        return "discard", "clipping"
    # ---- [ADD] 静音硬规则：整段静音比例 / 首尾静音 ----
    sil_frac = float(item.get("sil_frac", 0.0) or 0.0)
    lead_sil = float(item.get("lead_sil", 0.0) or 0.0)
    tail_sil = float(item.get("tail_sil", 0.0) or 0.0)
    if sil_frac > SIL_FRAC_MAX:
        return "discard", "too_much_silence"
    if (lead_sil > LEAD_TAIL_SIL_MAX) or (tail_sil > LEAD_TAIL_SIL_MAX):
        return "discard", "lead_or_tail_silence"
    # ---- [ADD] 时长硬边界：极短/极长直接丢弃 ----
    if dur < SHORT_FLOOR:
        return "discard", "ultra-short"
    if dur > LONG_CEIL:
        return "discard", "ultra_long"
    # ---- [MODIFY] 软边界（缓冲带）：短于 MIN_DUR / 长于 MAX_DUR → 参考库 ----
    if dur < min_dur:
        if cps is not None and cps > speed_fast and KEEP_SHORT_BUT_FAST_TO_TRAIN:
            return "keep_train", "too_short_but_fast"
        return "move_ref", ("too_short_but_fast" if (cps is not None and cps > speed_fast) else "too_short")
    if dur > max_dur:
        return "move_ref", "too_long"
    # ---- 正常时长：能量/F0 极端 or 语速极端 → 参考库 ----
    is_high_energy = rms >= rms_hi
    is_high_pitch  = (f0 > 0) and (f0 >= f0_hi)
    if is_high_energy and is_high_pitch:
        return "move_ref", "high_energy_high_f0"
    if cps is not None and cps > speed_fast:
        return "move_ref", "speed_fast"
    if cps is not None and cps < speed_slow:
        return "move_ref", "speed_slow"
    return "keep_train", "normal"


def split_and_write_csvs_by_actions(report_df: pd.DataFrame, orig_csv_path: Path,
                                    out_train_csv: Path, out_ref_csv: Path,
                                    out_discard_csv: Path | None = None):
    """
    用筛选报告里的 action(decide_action 的结果) 去拆原始 CSV：
    - keep_train → 进 train CSV
    - move_ref   → 进 ref CSV
    - discard    → 默认不进上述两份；若提供 out_discard_csv 则另存清单
    """
    if not orig_csv_path or not orig_csv_path.exists():
        print("[WARN] 未提供有效的 CSV_PATH，跳过 CSV 拆分。"); return

    # 读原 CSV（保留原列）
    try:
        dfc = pd.read_csv(orig_csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        dfc = pd.read_csv(orig_csv_path, encoding="utf-8-sig")

    # 找文件名列（尽量不改动原 CSV 的列名/内容）
    file_col = None
    for cand in ["文件名", "语音文件", "新文件名"]:
        if cand in dfc.columns:
            file_col = cand; break
    if file_col is None or "文本" not in dfc.columns:
        print("[WARN] 原 CSV 缺少必要列（文件名/语音文件/新文件名, 文本），跳过 CSV 拆分。")
        return

    # 从报告构建 basename → action 映射
    # 报告里存的是 rel_path（相对 WAV_DIR 的路径），需要取 basename 来对齐 CSV 中的文件名
    mp_action = {}
    for _, r in report_df.iterrows():
        base = os.path.basename(str(r["rel_path"]))
        mp_action[base] = r["action"]

    # 在原 CSV 上增加辅助列：basename、action（可能有些行找不到音频 -> action 为空）
    dfc["_base"] = dfc[file_col].astype(str).map(lambda s: os.path.basename(s.strip()))
    dfc["_action"] = dfc["_base"].map(mp_action)

    # 切分
    df_train   = dfc[dfc["_action"] == "keep_train"].drop(columns=["_base", "_action"])
    df_ref     = dfc[dfc["_action"].str.startswith("move_ref", na=False)].drop(columns=["_base", "_action"])
    df_discard = dfc[dfc["_action"] == "discard"].drop(columns=["_base", "_action"])

    # 写出（统一加引号，防止文本里逗号）
    out_train_csv.parent.mkdir(parents=True, exist_ok=True)
    out_ref_csv.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(out_train_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    df_ref.to_csv(out_ref_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    if out_discard_csv:
        out_discard_csv.parent.mkdir(parents=True, exist_ok=True)
        df_discard.to_csv(out_discard_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    # 日志
    print(f"[CSV] train_csv  : {len(df_train)} 行 -> {out_train_csv.resolve()}")
    print(f"[CSV] ref_csv    : {len(df_ref)} 行 -> {out_ref_csv.resolve()}")
    if out_discard_csv:
        print(f"[CSV] discard_csv: {len(df_discard)} 行 -> {out_discard_csv.resolve()}")
    else:
        print(f"[CSV] discard 已从 train/ref 中剔除（未单独另存）。")

# ============ 主流程（无需命令行参数）===========
def main():
    # 1) 选择目录
    wav_dir_str = WAV_DIR
    if (not wav_dir_str) and PICK_DIR_GUI:
        sel = pick_dir_dialog()
        if sel: wav_dir_str = sel
    if not wav_dir_str:
        print("[ERROR] 未指定 WAV 目录。请在脚本顶部设置 WAV_DIR，或开启 PICK_DIR_GUI 弹窗选择。")
        sys.exit(1)

    wav_dir = Path(wav_dir_str)
    if not wav_dir.exists():
        print(f"[ERROR] WAV 目录不存在：{wav_dir}")
        sys.exit(1)

    print(f"[INFO] 扫描目录: {wav_dir}")
    wavs = sorted([p for p in wav_dir.glob("**/*.wav")])
    print(f"[INFO] 找到 WAV 数量: {len(wavs)}")
    if not wavs:
        print("[WARN] 未找到任何 wav 文件，退出。"); return

    # 2) 文本映射（用于语速）
    text_map = {}
    if CSV_PATH:
        text_map = load_csv_text_map(Path(CSV_PATH))

    # 3) 扫描统计
    rows = []
    for idx, p in enumerate(wavs, 1):
        try:
            # ---- 一次性加载音频 ----
            y, sr = librosa.load(str(p), sr=SR, mono=True)
            dur = len(y) / sr

            # ---- 逐帧 RMS / 全局能量 ----
            rms_frames = rms_per_frame(y, FRAME_LENGTH, HOP_LENGTH)
            mean_rms = float(np.mean(rms_frames)) if rms_frames.size else 0.0

            # ---- F0 中位数 ----
            f0 = librosa.yin(y, fmin=50, fmax=600, sr=sr,
                             frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
            f0 = f0[np.isfinite(f0)]
            median_f0 = float(np.median(f0)) if f0.size else 0.0

            # ---- 静音指标（整段占比、首/尾静音时长）----
            sil_frac, lead_sec, tail_sec = silence_metrics(
                rms_frames, SR, HOP_LENGTH, SIL_RMS_THR
            )

            # ---- 削波检测 ----
            peak, near_frac, is_clip = detect_clipping(y)

            # ---- 文本 & 语速（只数中文）----
            base = p.name
            txt = text_map.get(base, "")
            n_chars = count_cn_chars(txt) if txt else 0
            cps = (n_chars / dur) if (dur > 0 and n_chars > 0) else None

            # ---- 写入行 ----
            rows.append({
                "rel_path": str(p.relative_to(wav_dir)),
                "duration": dur,
                "mean_rms": mean_rms,
                "median_f0": median_f0,
                "text_chars": n_chars,
                "cps": cps,
                "sil_frac": float(sil_frac),
                "lead_sil": float(lead_sec),
                "tail_sil": float(tail_sec),
                "clip_peak": float(peak),
                "clip_near_frac": float(near_frac),
                "is_clipped": bool(is_clip),
            })

            if idx % 50 == 0:
                print(f"[SCAN] {idx}/{len(wavs)} {base} | "
                      f"dur={dur:.2f}s rms={mean_rms:.6f} f0={median_f0:.1f} "
                      f"cps={cps if cps else 'NA'} sil={sil_frac:.2%} clip={is_clip}")
        except Exception as e:
            print(f"[ERR ] 读取失败: {p} -> {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] 无可用数据，退出。"); return

    # 4) 分位阈值（在正常时长范围内计算）
    mask_dur = (df["duration"] >= MIN_DUR) & (df["duration"] <= MAX_DUR)
    df_ok = df[mask_dur].copy()
    if df_ok.empty:
        print("[WARN] 正常时长范围内没有样本，阈值可能失真。")
        df_ok = df.copy()

    rms_hi = float(np.nanpercentile(df_ok["mean_rms"], PCTL_HIGH))
    f0_series = df_ok["median_f0"].replace(0, np.nan)
    f0_hi = float(np.nanpercentile(f0_series, PCTL_HIGH))

    print(f"[THR ] RMS_hi={rms_hi:.6f}  F0_hi={f0_hi:.2f}")
    print(f"[THR ] SPEED: slow<{SPEED_SLOW}  fast>{SPEED_FAST} (chars/sec)")

    # 5) 决策
    actions, reasons = [], []
    for _, r in df.iterrows():
        act, why = decide_action(r, rms_hi, f0_hi, MIN_DUR, MAX_DUR, SPEED_FAST, SPEED_SLOW)
        actions.append(act); reasons.append(why)
    df["action"] = actions
    df["reason"] = reasons

    # 6) 导出报告
    out_csv = Path(REPORT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[DONE] 报告已写出：{out_csv.resolve()}")

    # 7) 统计概览（数量+占比）
    total = len(df)
    def pct(n): return f"{(100.0*n/total):.2f}%"
    counts_action = df["action"].value_counts().to_dict()
    counts_reason = df["reason"].value_counts().to_dict()

    print("\n===== 概览（按 action）=====")
    for k, v in counts_action.items():
        print(f"{k:12s}: {v:5d}  ({pct(v)})")
    print("===== 概览（按 reason）=====")
    for k, v in counts_reason.items():
        print(f"{k:20s}: {v:5d}  ({pct(v)})")

    # 6.5) [ADD] 同步拆分原始 CSV（按 action）
    if CSV_PATH:
        split_and_write_csvs_by_actions(
            report_df=df,
            orig_csv_path=Path(CSV_PATH),
            out_train_csv=Path(TRAIN_CSV_OUT) if TRAIN_CSV_OUT else Path(REPORT_CSV).with_name("train.csv"),
            out_ref_csv=Path(REF_CSV_OUT) if REF_CSV_OUT else Path(REPORT_CSV).with_name("ref.csv"),
            out_discard_csv=(Path(DISCARD_CSV_OUT) if DISCARD_CSV_OUT else None)
        )

    # 8) 实际搬运（可选）
    if APPLY_MOVE:
        train_out = Path(TRAIN_OUT)
        ref_out = Path(REF_OUT)
        discard_out = Path(DISCARD_OUT) if 'DISCARD_OUT' in globals() and DISCARD_OUT else None  # [ADD]

        train_out.mkdir(parents=True, exist_ok=True)
        ref_out.mkdir(parents=True, exist_ok=True)
        if discard_out:  # [ADD]
            discard_out.mkdir(parents=True, exist_ok=True)

        mover = shutil.copy2 if USE_COPY else shutil.move

        kept = moved_ref = discarded = 0
        for i, r in df.iterrows():
            src = wav_dir / r["rel_path"]
            if not src.exists():
                print(f"[MISS] 源文件缺失: {src}"); continue
            act = r["action"]
            if act == "keep_train":
                dst = train_out / src.name
                if dst.exists(): os.remove(dst)
                mover(src, dst); kept += 1
            elif act.startswith("move_ref"):
                dst = ref_out / src.name
                if dst.exists(): os.remove(dst)
                mover(src, dst); moved_ref += 1
            else:  # discard
                if discard_out:  # [ADD] 若配置了丢弃目录，则搬过去
                    dst = discard_out / src.name
                    if dst.exists(): os.remove(dst)
                    mover(src, dst)
                discarded += 1

        print("\n===== 搬运结果 =====")
        print(f"keep_train: {kept}")
        print(f"move_ref  : {moved_ref}")
        print(f"discard   : {discarded}")
        print(f"[PATH] train_out = {train_out.resolve()}")
        print(f"[PATH] ref_out   = {ref_out.resolve()}")
        if discard_out:
            print(f"[PATH] discard_out = {discard_out.resolve()}")  # [ADD]
    else:
        print("\n[INFO] 仅生成报告（APPLY_MOVE=False）。如需实际分类，请在脚本顶部改成 True。")

if __name__ == "__main__":
    main()