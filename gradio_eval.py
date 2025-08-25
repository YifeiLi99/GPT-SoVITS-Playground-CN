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
    "REF_COLS": ["text", "ref", "ref_text", "gt", "target", "参考文本"],
    "ASR_A_COLS": ["asr_a", "hyp_a", "pred_a", "result_a", "transcript_a"],
    "ASR_B_COLS": ["asr_b", "hyp_b", "pred_b", "result_b", "transcript_b"],
    "ASR_COLS": ["asr", "hyp", "pred", "result", "transcript", "asr处理文本"],

    # 输出文件名（相对路径优先）
    "OUT_DIR_DEFAULT": "./eval_out",
    "OUT_DETAIL": "detail.csv",
    "OUT_SUMMARY": "summary.csv",
}

# =============================
# [ADD] 轻量 Levenshtein（可选 rapidfuzz 加速）
# =============================
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
        res = compare_two(stdA, stdB)
    else:
        std = standardize_csv(csvA or csvB)
        res = compare_single(std)

    detail_path, summary_path = _write_outputs(res, out_dir)
    report = _summary_text(res)

    # 控制台输出
    print("\n===== SUMMARY =====")
    print(report)
    print("\n输出：")
    print(" - 明细:", os.path.relpath(detail_path))
    print(" - 汇总:", os.path.relpath(summary_path))

    return detail_path, summary_path, report


# =============================
# [ADD] Gradio UI（对齐 gradio_asr 风格）
# =============================

def run_ui():
    try:
        import gradio as gr
    except Exception:
        print("[ERROR] 未安装 gradio；请先 `pip install gradio`，或使用 --ui 0 走 CLI")
        return

    CONTROL_URL = CONFIG["CONTROL_URL"]

    def _go_eval(csv_a_path, csv_b_path, out_dir):
        args = _Args(csv_a_path.strip() or "", csv_b_path.strip() or "", out_dir.strip() or CONFIG["OUT_DIR_DEFAULT"])
        try:
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

        # ===== 输出区：报告文本 + 下载文件 =====
        report_out = gr.Textbox(label="对比汇总报告", lines=8)
        out_detail = gr.File(label="下载 detail.csv")
        out_summary = gr.File(label="下载 summary.csv")

        btn_run.click(_go_eval, [csv_a_path, csv_b_path, out_dir], [report_out, out_detail, out_summary])

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
