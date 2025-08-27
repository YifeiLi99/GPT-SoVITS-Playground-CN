#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 评测 / 对比（CSV 最小核心版，整合入主控台风格）
=================================================
变更总览（与你现有 gradio_eval.py 一致核心、对齐 gradio_asr 风格）：
- [ADD] 顶部导航按钮（TTS/结果评测/微调/数据清洗/语音桥接/主控台），与 gradio_asr 同步；
- [CHG] Gradio UI 改为 **3 个输入框**（CSV_A 路径、CSV_B 路径、输出目录）+ **一键开始对比**；
- [ADD] 结果展示区：
    1) 文本框展示本次汇总报告（均值 CER、对齐样本数、模式等）；
    2) 自动写出 detail.csv / summary.csv（默认相对路径 ./eval_out/）；页面下方可直接下载；
- [CHG] 全流程默认使用 **相对路径** 输出，符合你的长期要求；
- [CHG] CSV 写出统一使用 `csv.QUOTE_ALL`，避免未加引号的风险；
- [KEEP] CLI 功能保持，可 `--ui 0` 走命令行；
- [TODO] 后续可把 CONFIG 挪入 config.py（已集中在顶部）。

依赖：numpy, pandas, tqdm（可选 rapidfuzz）; gradio（v3/4 兼容）。
"""

from __future__ import annotations
import os
import csv
import argparse
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================
# [ADD] 统一配置（后续可迁移到 config.py）
# =============================
CONFIG = {
    # 界面与端口
    "LANGUAGE": "zh",
    "DEFAULT_PORT": 9881,
    "GRADIO_QUEUE": True,
    "GRADIO_SERVER_NAME": "0.0.0.0",
    # 顶部主控台地址（与 gradio_asr.py 一致）
    "CONTROL_URL": "http://127.0.0.1:9900/",

    # 列名候选（不区分大小写）
    "KEY_COLS": ["utt_id", "id", "key", "sid", "uid"],
    "WAV_COLS": ["wav", "wav_path", "path", "audio", "audio_path", "语音文件", "文件路径"],
    "REF_COLS": ["text", "ref", "ref_text", "gt", "target", "参考文本", "原始文本"],
    "ASR_A_COLS": ["asr_a", "hyp_a", "pred_a", "result_a", "transcript_a"],
    "ASR_B_COLS": ["asr_b", "hyp_b", "pred_b", "result_b", "transcript_b"],
    "ASR_COLS": ["asr", "hyp", "pred", "result", "transcript", "asr处理文本"],

    # 输出文件名（相对路径优先）
    "OUT_DIR_DEFAULT": "./eval_out",
    "OUT_DETAIL": "detail.csv",
    "OUT_SUMMARY": "summary.csv",
    "NORM_ENABLE": True,
    "NORM_STRIP_PUNCT": True,
    "NORM_STRIP_SPACES": True,
    "NORM_STRIP_INTERJ": True,
    "NORM_TO_SIMPLIFIED": True,  # 若系统未安装 opencc，则自动回退为 False
    "NORM_CACHE_DIR": "./eval_out/_norm_cache",

    "LONG_SENT_THRES": 12,
    "PUNC_FOR_PROSODY": "，,。.!！?？、…；;：:",
    "KEYWORDS_DEFAULT": ["哇","啊","嗯","嘿嘿","哈哈"],
    "TARGET_CHARS_PER_SEC": 6.0,  # 目标语速（字符/秒），可根据语料调整

}

# =============================
# [ADD] 轻量 Levenshtein（可选 rapidfuzz 加速）
# =============================
# =============================
# [ADD] 评测指标实现（CER/WER/关键词/吞字/韵律/语速）
# =============================

# [ADD] 分词/标记：中文无空格时按“中英文混合”策略切分；英文/带空格用空格切分
_word_re = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\s]")

def tokenize_for_wer(s: str):
    s = (s or "").strip()
    if not s:
        return []
    # 如果有空格，优先按空格（英文/带空格语料）
    if " " in s.strip():
        return s.split()
    # 否则按“英文串/单个中文汉字/其他单字符”混合切分
    return _word_re.findall(s)

def wer_word(ref: str, hyp: str) -> float:
    R = tokenize_for_wer(ref)
    H = tokenize_for_wer(hyp)
    if len(R) == 0:
        return 0.0 if len(H) == 0 else 1.0
    # 经典 Levenshtein on tokens
    n, m = len(R), len(H)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if R[i-1] == H[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,     # deletion
                           dp[i][j-1] + 1,     # insertion
                           dp[i-1][j-1] + cost)  # substitution
    return dp[n][m] / max(1, n)

# [ADD] 编辑距离操作统计（用于“吞字率”=删除比例）
def edit_ops(ref: str, hyp: str):
    R = list(ref or "")
    H = list(hyp or "")
    n, m = len(R), len(H)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
        if i>0: bt[i][0] = ('D', i-1, 0)
    for j in range(m+1):
        dp[0][j] = j
        if j>0: bt[0][j] = ('I', 0, j-1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if R[i-1] == H[j-1]:
                dp[i][j] = dp[i-1][j-1]
                bt[i][j] = ('M', i-1, j-1)
            else:
                # choose best among D, I, S
                cand = [
                    (dp[i-1][j] + 1, 'D', i-1, j),    # deletion
                    (dp[i][j-1] + 1, 'I', i, j-1),    # insertion
                    (dp[i-1][j-1] + 1, 'S', i-1, j-1) # substitution
                ]
                best = min(cand, key=lambda x:x[0])
                dp[i][j] = best[0]
                bt[i][j] = (best[1], best[2], best[3])
    # backtrace
    i, j = n, m
    dels = ins = subs = match = 0
    while i>0 or j>0:
        op, pi, pj = bt[i][j] if bt[i][j] else ('M', i-1, j-1)
        if op == 'D': dels += 1
        elif op == 'I': ins += 1
        elif op == 'S': subs += 1
        else: match += 1
        i, j = pi, pj
    return {'del':dels, 'ins':ins, 'sub':subs, 'match':match, 'dist':dp[n][m], 'n_ref':n, 'n_hyp':m}

# [ADD] 吞字率（长句）：删除比例，仅在 ref_len >= 阈值 时计入
def swallow_rate(ref: str, hyp: str, long_thres: int = 20) -> Optional[float]:
    ref = ref or ""
    hyp = hyp or ""
    if len(ref) < long_thres:
        return None
    ops = edit_ops(ref, hyp)
    return ops['del'] / max(1, ops['n_ref'])

# [ADD] 基于标点的“韵律抖动方差”（代理指标）：
#      以标点分段，计算“每段长度差（hyp-ref）”的方差/均方误差
PUNC_DEFAULT = "，,。.!！?？、…；;：:"

def _segments_by_punc(s: str, punc: str) -> list[int]:
    buf = 0
    arr = []
    for ch in s or "":
        if ch in punc:
            arr.append(buf)
            buf = 0
        else:
            buf += 1
    arr.append(buf)
    return arr

def prosody_jitter_var(ref: str, hyp: str, punc: str = PUNC_DEFAULT) -> float:
    a = _segments_by_punc(ref or "", punc)
    b = _segments_by_punc(hyp or "", punc)
    # 对齐较短长度
    L = min(len(a), len(b)) if a and b else 0
    if L == 0:
        return 0.0
    diffs = [(b[i]-a[i]) for i in range(L)]
    # 归一化到 ref 段长，避免长度规模影响
    norm = [ (diff / (a[i] if a[i]>0 else 1)) for i, diff in enumerate(diffs) ]
    # 返回均方（方差代理）
    return float(np.mean([x*x for x in norm]))

# [ADD] 关键词一致性：计算 ref→hyp 的“召回率”，以及 hyp 中“额外关键词”比率
def keyword_metrics(ref: str, hyp: str, keywords: List[str]) -> Tuple[Optional[float], Optional[float]]:
    kws = [k for k in (keywords or []) if k]
    if not kws:
        return None, None
    ref_hits = sum(1 for k in kws if k in (ref or ""))
    hyp_hits = sum(1 for k in kws if k in (hyp or ""))
    recall = (hyp_hits / ref_hits) if ref_hits>0 else (None)
    # 额外关键词：hyp 中出现但 ref 中未出现的占比
    extra = sum(1 for k in kws if (k in (hyp or "")) and (k not in (ref or "")))
    extra_ratio = (extra / len(kws)) if kws else None
    return recall, extra_ratio

# [ADD] 语速误差：需要 wav 时长；若拿不到 wav 则返回 None
def _wav_duration_seconds(path: str) -> Optional[float]:
    try:
        import soundfile as sf
        import numpy as np  # noqa: F401
        with sf.SoundFile(path) as f:
            frames = len(f)
            sr = f.samplerate
        return frames / float(sr)
    except Exception:
        try:
            import wave
            with wave.open(path, 'rb') as wf:
                frames = wf.getnframes()
                sr = wf.getframerate()
            return frames / float(sr)
        except Exception:
            return None

def speed_error(ref: str, wav_path: Optional[str], target_chars_per_sec: Optional[float]) -> Optional[float]:
    if not wav_path or not target_chars_per_sec or target_chars_per_sec <= 0:
        return None
    dur = _wav_duration_seconds(wav_path)
    if not dur or dur <= 0:
        return None
    cps = (len(ref or "")) / dur
    return abs(cps - target_chars_per_sec) / target_chars_per_sec  # 绝对误差比


HAVE_RAPIDFUZZ = False
try:
    from rapidfuzz.distance import Levenshtein  # type: ignore
    HAVE_RAPIDFUZZ = True
except Exception:
    pass


def _levenshtein_py(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def cer_char(ref: str, hyp: str) -> float:
    ref = ref or ""
    hyp = hyp or ""
    if ref == "":
        return 0.0 if hyp == "" else 1.0
    if HAVE_RAPIDFUZZ:
        dist = Levenshtein.distance(ref, hyp)
    else:
        dist = _levenshtein_py(ref, hyp)
    return dist / max(1, len(ref))

# =============================
# [ADD] CSV 解析 & 列规范化
# =============================

# =============================
# [ADD] 文本标准化（简繁/标点/空白/语气字/NFKC）
# =============================
def _try_opencc():
    try:
        from opencc import OpenCC  # type: ignore
        return OpenCC('t2s')
    except Exception:
        return None

_OPENCC = _try_opencc()
if _OPENCC is None:
    CONFIG["NORM_TO_SIMPLIFIED"] = False  # 回退：未安装 opencc 则不做简繁转换

_INTERJ_SET = set(list("哇啊哦喔噢欸诶呃嗯唔呀嘿哈呵嘻哼呜"))  # 可再扩充

import unicodedata

def _nfkc(s: str) -> str:
    try:
        return unicodedata.normalize("NFKC", s or "")
    except Exception:
        return s or ""

def _to_simplified(s: str) -> str:
    if CONFIG.get("NORM_TO_SIMPLIFIED", False) and _OPENCC is not None:
        try:
            return _OPENCC.convert(s)
        except Exception:
            return s
    return s

def _strip_punct_and_spaces(s: str) -> str:
    # 去所有 Unicode 标点与空白
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P") and not ch.isspace())

def _strip_interjections(s: str) -> str:
    # 简单逐字剔除常见语气字（工程取舍：避免把“内容词”误删）
    return "".join(ch for ch in s if ch not in _INTERJ_SET)

def normalize_text(s: str) -> str:
    if s is None:
        s = ""
    s = str(s)
    s = _nfkc(s)
    s = _to_simplified(s)
    if CONFIG.get("NORM_STRIP_PUNCT", True) or CONFIG.get("NORM_STRIP_SPACES", True):
        s = _strip_punct_and_spaces(s)
    if CONFIG.get("NORM_STRIP_INTERJ", True):
        s = _strip_interjections(s)
    return s

def normalize_dataframe(std: StdFrame, tag: str) -> StdFrame:
    """
    返回新的 StdFrame，df 文本列已标准化；并把标准化后的 CSV 存到 NORM_CACHE_DIR 下。
    仅处理：ref/asr/asr_a/asr_b（其余列保持不动）。
    """
    import os, pandas as pd, csv
    df = std.df.copy()
    cols = [c for c in [std.ref, std.asr, std.asr_a, std.asr_b] if c is not None]
    if cols and CONFIG.get("NORM_ENABLE", True):
        for c in cols:
            df[c] = df[c].astype(str).map(normalize_text)
    # 写入缓存
    out_dir = CONFIG.get("NORM_CACHE_DIR", "./eval_out/_norm_cache")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_normalized.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    # 返回新的 StdFrame
    std2 = StdFrame(df=df, key=std.key, wav=std.wav, ref=std.ref, asr=std.asr, asr_a=std.asr_a, asr_b=std.asr_b)
    return std2

LOWER = lambda s: (s or "").strip().lower()


def _find_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    lower_map = {LOWER(c): c for c in df.columns}
    for k in cands:
        if LOWER(k) in lower_map:
            return lower_map[LOWER(k)]
    return None


@dataclass
class StdFrame:
    df: pd.DataFrame
    key: Optional[str]
    wav: Optional[str]
    ref: Optional[str]
    asr: Optional[str]
    asr_a: Optional[str]
    asr_b: Optional[str]


def standardize_csv(path: str) -> StdFrame:
    # 容错读取编码
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")

    key = _find_col(df, CONFIG["KEY_COLS"]) or None
    wav = _find_col(df, CONFIG["WAV_COLS"]) or None
    ref = _find_col(df, CONFIG["REF_COLS"]) or None
    asr = _find_col(df, CONFIG["ASR_COLS"]) or None
    asr_a = _find_col(df, CONFIG["ASR_A_COLS"]) or None
    asr_b = _find_col(df, CONFIG["ASR_B_COLS"]) or None
    return StdFrame(df=df, key=key, wav=wav, ref=ref, asr=asr, asr_a=asr_a, asr_b=asr_b)


# =============================
# [ADD] 对齐策略：绝对 wav 路径 > key > 文件名 > 行号
# =============================

def _norm_path(p: str) -> str:
    try:
        return os.path.normpath(os.path.abspath(p))
    except Exception:
        return (p or "").strip()


def build_join_key(std: StdFrame) -> pd.Series:
    df = std.df
    key_series = None

    if std.wav is not None:
        key_series = df[std.wav].astype(str).map(_norm_path)
    if key_series is None and std.key is not None:
        key_series = df[std.key].astype(str)
    if key_series is None and std.wav is not None:
        key_series = df[std.wav].astype(str).map(lambda x: os.path.basename(str(x)))
    if key_series is None:
        key_series = pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)
    return key_series

# =============================
# [ADD] 三种对比模式
# =============================
@dataclass
class CompareResult:
    detail: pd.DataFrame
    summary: pd.DataFrame


def compare_single(std: StdFrame) -> CompareResult:
    df = std.df.copy()
    join_key = build_join_key(std)

    # 优先：ASR_A vs ASR_B
    if std.asr_a is not None and std.asr_b is not None:
        A = df[std.asr_a].astype(str)
        B = df[std.asr_b].astype(str)
        cer_ab = [cer_char(a, b) for a, b in zip(A, B)]
        detail = pd.DataFrame({
            "key": join_key,
            "asr_A": A,
            "asr_B": B,
            "CER_A_vs_B": np.round(cer_ab, 4),
        })
        summary = pd.DataFrame([{
            "mode": "single_csv_asrA_vs_asrB",
            "n": len(detail),
            "CER_A_vs_B_mean": float(np.mean(detail["CER_A_vs_B"])) if len(detail) else np.nan,
        }])
        return CompareResult(detail, summary)

    # 其次：ASR vs REF
    if std.asr is not None and std.ref is not None:
        H = df[std.asr].astype(str)
        R = df[std.ref].astype(str)
        cer_hr = [cer_char(r, h) for r, h in zip(R, H)]
        detail = pd.DataFrame({
            "key": join_key,
            "ref": R,
            "asr": H,
            "CER_ASR_vs_REF": np.round(cer_hr, 4),
        })
        summary = pd.DataFrame([{
            "mode": "single_csv_asr_vs_ref",
            "n": len(detail),
            "CER_ASR_vs_REF_mean": float(np.mean(detail["CER_ASR_vs_REF"])) if len(detail) else np.nan,
        }])
        return CompareResult(detail, summary)

    raise ValueError("单CSV对比失败：未检测到 (asr_A, asr_B) 或 (asr, ref) 列")


def compare_two(stdA: StdFrame, stdB: StdFrame) -> CompareResult:
    # [KEEP] 原有 A/B 文本直接 CER 对比（非常用）

    dfA = stdA.df.copy()
    dfB = stdB.df.copy()
    keyA = build_join_key(stdA)
    keyB = build_join_key(stdB)

    def pick_text(std: StdFrame) -> Tuple[Optional[str], str]:
        if std.asr is not None:
            return std.asr, "asr"
        if std.asr_a is not None:
            return std.asr_a, "asr_a"
        if std.asr_b is not None:
            return std.asr_b, "asr_b"
        if std.ref is not None:
            return std.ref, "ref_as_fallback"
        return None, ""

    colA, tagA = pick_text(stdA)
    colB, tagB = pick_text(stdB)
    if colA is None or colB is None:
        raise ValueError("双CSV对比失败：至少一侧未找到可用文本列(asr/asr_a/asr_b/ref)")

    A = pd.DataFrame({"key": keyA, "text_A": dfA[colA].astype(str)})
    B = pd.DataFrame({"key": keyB, "text_B": dfB[colB].astype(str)})

    merged = A.merge(B, on="key", how="inner")
    if merged.empty:
        raise ValueError("双CSV对比失败：两侧按 key 无可对齐样本（检查 wav 路径/utt_id）")

    cer_ab = [cer_char(a, b) for a, b in zip(merged["text_A"], merged["text_B"])]
    detail = pd.DataFrame({
        "key": merged["key"],
        "text_A": merged["text_A"],
        "text_B": merged["text_B"],
        "CER_A_vs_B": np.round(cer_ab, 4),
        "A_source": tagA,
        "B_source": tagB,
    })

    summary = pd.DataFrame([{
        "mode": "two_csv_asr_vs_asr",
        "n_join": int(len(detail)),
        "CER_A_vs_B_mean": float(np.mean(detail["CER_A_vs_B"])) if len(detail) else np.nan,
        "A_source": tagA,
        "B_source": tagB,
    }])
    return CompareResult(detail, summary)


# [ADD] 扩展：双 CSV 的完整指标（以 A.ref 优先作为基准；若 B.ref 存在且一致则无影响）
def compare_two_full(stdA: StdFrame, stdB: StdFrame, long_thres: int, punc: str, keywords: List[str], target_cps: Optional[float]) -> CompareResult:
    dfA = stdA.df.copy()
    dfB = stdB.df.copy()
    keyA = build_join_key(stdA)
    keyB = build_join_key(stdB)

    # 选择文本列
    if stdA.asr is None or stdB.asr is None:
        raise ValueError("双CSV完整指标：两侧都需存在 ASR 列（如 'asr' 或 'asr处理文本'）")

    if stdA.ref is None and stdB.ref is None:
        raise ValueError("双CSV完整指标：至少一侧需提供 REF 列（如 'text'/'原始文本'）")

    # 合并对齐
    A = pd.DataFrame({"key": keyA, "ref_A": dfA[stdA.ref] if stdA.ref else None, "asr_A": dfA[stdA.asr].astype(str), "wav_A": dfA[stdA.wav] if stdA.wav else None})
    B = pd.DataFrame({"key": keyB, "ref_B": dfB[stdB.ref] if stdB.ref else None, "asr_B": dfB[stdB.asr].astype(str), "wav_B": dfB[stdB.wav] if stdB.wav else None})
    merged = A.merge(B, on="key", how="inner")

    if merged.empty:
        raise ValueError("双CSV完整指标：按 key 无对齐样本（检查 '文件地址/utt_id/文件名' 等列）")

    # 基准 REF：优先 A 的 ref；若缺失则用 B 的 ref
    def choose_ref(row):
        return (row["ref_A"] if pd.notna(row.get("ref_A", None)) else row.get("ref_B", "")) or ""

    rows = []
    for _, r in merged.iterrows():
        ref = str(choose_ref(r))
        asrA = str(r["asr_A"])
        asrB = str(r["asr_B"])
        wavA = str(r["wav_A"]) if pd.notna(r.get("wav_A")) else None
        wavB = str(r["wav_B"]) if pd.notna(r.get("wav_B")) else None

        cer_A = cer_char(ref, asrA)
        cer_B = cer_char(ref, asrB)
        wer_A = wer_word(ref, asrA)
        wer_B = wer_word(ref, asrB)

        sr_A = swallow_rate(ref, asrA, long_thres)
        sr_B = swallow_rate(ref, asrB, long_thres)

        pjv_A = prosody_jitter_var(ref, asrA, punc)
        pjv_B = prosody_jitter_var(ref, asrB, punc)

        kw_rec_A, kw_extra_A = keyword_metrics(ref, asrA, keywords)
        kw_rec_B, kw_extra_B = keyword_metrics(ref, asrB, keywords)

        se_A = speed_error(ref, wavA, target_cps)
        se_B = speed_error(ref, wavB, target_cps)

        rows.append({
            "key": r["key"],
            "ref": ref,
            "asr_A": asrA,
            "asr_B": asrB,
            "CER_A": round(cer_A, 4),
            "CER_B": round(cer_B, 4),
            "WER_A": round(wer_A, 4),
            "WER_B": round(wer_B, 4),
            "Swallow_A": (round(sr_A,4) if sr_A is not None else None),
            "Swallow_B": (round(sr_B,4) if sr_B is not None else None),
            "ProsodyVar_A": round(pjv_A, 6),
            "ProsodyVar_B": round(pjv_B, 6),
            "KW_Recall_A": (round(kw_rec_A,4) if kw_rec_A is not None else None),
            "KW_Recall_B": (round(kw_rec_B,4) if kw_rec_B is not None else None),
            "KW_Extra_A": (round(kw_extra_A,4) if kw_extra_A is not None else None),
            "KW_Extra_B": (round(kw_extra_B,4) if kw_extra_B is not None else None),
            "SpeedErr_A": (round(se_A,4) if se_A is not None else None),
            "SpeedErr_B": (round(se_B,4) if se_B is not None else None),
        })

    detail = pd.DataFrame(rows)
    # 只在非空时计算均值
    def _mean(series):
        try:
            return float(pd.to_numeric(series, errors="coerce").dropna().mean())
        except Exception:
            return float("nan")

    summary = pd.DataFrame([{
        "mode": "two_csv_metrics_vs_ref",
        "n_join": int(len(detail)),
        "CER_mean_A": _mean(detail["CER_A"]),
        "CER_mean_B": _mean(detail["CER_B"]),
        "WER_mean_A": _mean(detail["WER_A"]),
        "WER_mean_B": _mean(detail["WER_B"]),
        "Swallow_mean_A": _mean(detail["Swallow_A"]),
        "Swallow_mean_B": _mean(detail["Swallow_B"]),
        "ProsodyVar_mean_A": _mean(detail["ProsodyVar_A"]),
        "ProsodyVar_mean_B": _mean(detail["ProsodyVar_B"]),
        "KW_Recall_mean_A": _mean(detail["KW_Recall_A"]),
        "KW_Recall_mean_B": _mean(detail["KW_Recall_B"]),
        "KW_Extra_mean_A": _mean(detail["KW_Extra_A"]),
        "KW_Extra_mean_B": _mean(detail["KW_Extra_B"]),
        "SpeedErr_mean_A": _mean(detail["SpeedErr_A"]),
        "SpeedErr_mean_B": _mean(detail["SpeedErr_B"]),
        # A/B 差值（B-A，正数= B 更大，负数= B 更小/更好）
        "CER_delta_B_minus_A": _mean(detail["CER_B"]) - _mean(detail["CER_A"]),
        "WER_delta_B_minus_A": _mean(detail["WER_B"]) - _mean(detail["WER_A"]),
        "Swallow_delta_B_minus_A": _mean(detail["Swallow_B"]) - _mean(detail["Swallow_A"]),
        "ProsodyVar_delta_B_minus_A": _mean(detail["ProsodyVar_B"]) - _mean(detail["ProsodyVar_A"]),
        "KW_Recall_delta_B_minus_A": _mean(detail["KW_Recall_B"]) - _mean(detail["KW_Recall_A"]),
        "KW_Extra_delta_B_minus_A": _mean(detail["KW_Extra_B"]) - _mean(detail["KW_Extra_A"]),
        "SpeedErr_delta_B_minus_A": _mean(detail["SpeedErr_B"]) - _mean(detail["SpeedErr_A"]),
    }])
    return CompareResult(detail, summary)

# =============================
# [ADD] CLI / UI 共用入口
# =============================

def _write_outputs(res: CompareResult, out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, CONFIG["OUT_DETAIL"])  # 相对路径输出
    summary_path = os.path.join(out_dir, CONFIG["OUT_SUMMARY"])  # 相对路径输出
    # 统一加引号写出
    res.detail.to_csv(detail_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    res.summary.to_csv(summary_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    return detail_path, summary_path


def _summary_text(res: CompareResult) -> str:
    # 紧凑可读的报告摘要（放在 UI 文本框里）
    s = []
    mode = str(res.summary.iloc[0].get("mode", "-") if len(res.summary) else "-")
    s.append(f"[MODE] {mode}")
    for col in res.summary.columns:
        if col == "mode":
            continue
        val = res.summary.iloc[0].get(col)
        s.append(f"{col}: {val}")
    return "\n".join(s)


class _Args:  # 简易命名空间
    def __init__(self, csv_a: Optional[str], csv_b: Optional[str], out_dir: Optional[str]):
        self.csv_a = csv_a
        self.csv_b = csv_b
        self.out_dir = out_dir or CONFIG["OUT_DIR_DEFAULT"]


def run_cli(args) -> Tuple[str, str, str]:
    out_dir = args.out_dir or CONFIG["OUT_DIR_DEFAULT"]
    csvA = (args.csv_a or "").strip()
    csvB = (args.csv_b or "").strip()

    if (not csvA) and (not csvB):
        raise ValueError("请至少提供 --csv_a 或同时提供 --csv_a --csv_b")
    if csvA and (not os.path.isfile(csvA)):
        raise FileNotFoundError(f"csv_a 不存在：{csvA}")
    if csvB and (not os.path.isfile(csvB)):
        raise FileNotFoundError(f"csv_b 不存在：{csvB}")

    if csvA and csvB:
        stdA = standardize_csv(csvA)
        stdB = standardize_csv(csvB)
        # [ADD] 比较前做标准化并写缓存 CSV
        stdA = normalize_dataframe(stdA, 'A')
        stdB = normalize_dataframe(stdB, 'B')
        try:
            res = compare_two_full(stdA, stdB, CONFIG['LONG_SENT_THRES'], CONFIG['PUNC_FOR_PROSODY'], CONFIG['KEYWORDS_DEFAULT'], CONFIG['TARGET_CHARS_PER_SEC'])
        except Exception as _:
            # 回退到原有简单 CER 对比
            res = compare_two(stdA, stdB)
    else:
        std = standardize_csv(csvA or csvB)
        std = normalize_dataframe(std, 'SINGLE')
        res = compare_single(std)

    detail_path, summary_path = _write_outputs(res, out_dir)
    report = _summary_text(res)

    # 控制台输出
    print("\n===== SUMMARY =====")
    print(report)
    print("\n输出：")
    print(f" - 标准化缓存: {CONFIG.get('NORM_CACHE_DIR','./eval_out/_norm_cache')}")
    print(" - 明细:", os.path.relpath(detail_path))
    print(" - 汇总:", os.path.relpath(summary_path))

    return detail_path, summary_path, report


# =============================
# [ADD] Gradio UI（对齐 gradio_asr 风格）
# =============================

def run_ui():
    # [ADD] UI 扩展：关键词 / 长句阈值 / 标点集 / 目标语速
    try:
        import gradio as gr
    except Exception:
        print("[ERROR] 未安装 gradio；请先 `pip install gradio`，或使用 --ui 0 走 CLI")
        return

    CONTROL_URL = CONFIG["CONTROL_URL"]

    def _go_eval(csv_a_path, csv_b_path, out_dir, keywords, long_thres, punc_set, target_cps):
        # 同时支持双CSV完整指标
        args = _Args(csv_a_path.strip() or "", csv_b_path.strip() or "", out_dir.strip() or CONFIG["OUT_DIR_DEFAULT"])
        try:
            if (csv_a_path.strip() and csv_b_path.strip()):
                stdA = standardize_csv(csv_a_path.strip())
                stdB = standardize_csv(csv_b_path.strip())
                stdA = normalize_dataframe(stdA, 'A')
                stdB = normalize_dataframe(stdB, 'B')
                kws = [k.strip() for k in (keywords or '').split(',') if k.strip()] or CONFIG['KEYWORDS_DEFAULT']
                res = compare_two_full(stdA, stdB, int(long_thres), (punc_set or CONFIG['PUNC_FOR_PROSODY']), kws, (float(target_cps) if target_cps else CONFIG['TARGET_CHARS_PER_SEC']))
                d, s = _write_outputs(res, args.out_dir)
                rep = _summary_text(res)
                return rep, d, s
            else:
                d, s, rep = run_cli(args)
                return rep, d, s
        except Exception as e:
            return f"[ERROR] {e}", None, None

    with gr.Blocks(title="CSV 对比 / 评测（最小版）") as demo:
        gr.Markdown("# 📊 CSV 对比 / 评测（最小版）\n上传路径不是必须，**直接填路径更高效**。支持：双CSV、单CSV(双ASR列)、单CSV(ASR vs REF)。")

        # ===== 顶部导航（与 gradio_asr 一致） =====
        with gr.Row():
            btn_go_tts = gr.Button("TTS", variant="secondary")
            btn_go_eval = gr.Button("结果评测", variant="secondary")
            btn_go_tune = gr.Button("微调", variant="secondary")
            btn_go_clean = gr.Button("数据清洗", variant="secondary")
            btn_go_bridge = gr.Button("语音桥接", variant="secondary")
            btn_go_home = gr.Button("主控台", variant="secondary")
        _js_to_tts = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=TTS'; return []; }}"
        _js_to_eval = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=EVAL'; return []; }}"
        _js_to_tune = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=FINETUNE'; return []; }}"
        _js_to_clean = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=CLEAN'; return []; }}"
        _js_to_bridge = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=BRIDGE'; return []; }}"
        _js_to_home = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}'; return []; }}"
        try:  # Gradio v4+
            btn_go_tts.click(None, [], [], js=_js_to_tts)
            btn_go_eval.click(None, [], [], js=_js_to_eval)
            btn_go_tune.click(None, [], [], js=_js_to_tune)
            btn_go_clean.click(None, [], [], js=_js_to_clean)
            btn_go_bridge.click(None, [], [], js=_js_to_bridge)
            btn_go_home.click(None, [], [], js=_js_to_home)
        except TypeError:  # 兼容 v3.x
            btn_go_tts.click(None, [], [], _js=_js_to_tts)
            btn_go_eval.click(None, [], [], _js=_js_to_eval)
            btn_go_tune.click(None, [], [], _js=_js_to_tune)
            btn_go_clean.click(None, [], [], _js=_js_to_clean)
            btn_go_bridge.click(None, [], [], _js=_js_to_bridge)
            btn_go_home.click(None, [], [], _js=_js_to_home)

        # ===== 3 个输入框 + 一键对比 =====
        with gr.Row():
            csv_a_path = gr.Textbox(label="CSV A 路径（单CSV时仅填 A）", placeholder=r"D:\path\to\A.csv")
            csv_b_path = gr.Textbox(label="CSV B 路径（可选）", placeholder=r"D:\path\to\B.csv")
            out_dir = gr.Textbox(label="输出目录（相对路径优先）", value=CONFIG["OUT_DIR_DEFAULT"])
        btn_run = gr.Button("开始对比", variant="primary")
        gr.Markdown("**说明**：双CSV模式将基于 `A/B.ASR` 对 `REF` 进行全指标评测；若读取不到 WAV 时长，则语速误差自动跳过。")

        # [ADD] 缺失的四个输入组件（用于 _go_eval 的参数）
        with gr.Row():
            keywords = gr.Textbox(label="关键词列表（逗号分隔）", value=",".join(CONFIG["KEYWORDS_DEFAULT"]))
            long_thres = gr.Number(label="长句阈值（字数）", value=CONFIG["LONG_SENT_THRES"])
            punc_set = gr.Textbox(label="标点集合（韵律代理）", value=CONFIG["PUNC_FOR_PROSODY"])
            target_cps = gr.Number(label="目标语速（字符/秒）", value=CONFIG["TARGET_CHARS_PER_SEC"])

        # ===== 输出区：报告文本 + 下载文件 =====
        report_out = gr.Textbox(label="对比汇总报告", lines=8)
        out_detail = gr.File(label="下载 detail.csv")
        out_summary = gr.File(label="下载 summary.csv")

        btn_run.click(_go_eval, [csv_a_path, csv_b_path, out_dir, keywords, long_thres, punc_set, target_cps], [report_out, out_detail, out_summary])

    # 启动参数（与 asr 对齐）
    if CONFIG["GRADIO_QUEUE"]:
        demo.queue(api_open=False, max_size=32)
    demo.launch(
        server_name=CONFIG["GRADIO_SERVER_NAME"],
        server_port=CONFIG["DEFAULT_PORT"],
        share=False,
        inbrowser=False,
    )


# =============================
# [ADD] 入口参数
# =============================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv_a', type=str, default='', help='CSV A 文件路径（单 CSV 模式仅需 A）')
    p.add_argument('--csv_b', type=str, default='', help='CSV B 文件路径（可选）')
    p.add_argument('--out_dir', type=str, default=CONFIG["OUT_DIR_DEFAULT"], help='输出目录（相对路径优先）')
    p.add_argument('--ui', type=int, default=1, help='1=打开UI，0=命令行')
    args = p.parse_args()

    if int(args.ui) == 1:
        run_ui()
    else:
        run_cli(args)
