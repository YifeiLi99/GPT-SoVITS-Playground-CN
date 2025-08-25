# -*- coding: utf-8 -*-
"""
normalize_tts_texts.py  —  带页面的 TTS 文本规范化工具（不走命令行）
用法：直接 `python normalize_tts_texts.py` 启动一个本地页面；页面里固定两个“绝对路径”，可手动修改。
依赖：pandas, gradio
"""

import re
import unicodedata
from pathlib import Path
import pandas as pd
import csv

# ===== 1) 语气词 / 口癖 归一（结合社区常见做法） =====
# 说明：合并为少量“标准写法”，减少发音分裂，便于建模
INTERJ_MAP = {
    # “嗯”类
    "唔": "嗯", "呣": "嗯",
    # “呃/额”类
    "额，": "呃,","呃啊": "呃","额,": "呃,",
    # “欸/诶”类
    "诶": "欸","呜欸": "欸","欸欸": "欸",
    # “啊/呀”类
    #"啊呀呀": "啊", "啊呀": "啊",
    "啊啊": "啊",
    # “哇/呜哇”类
    "呜哇": "哇","呜呜": "呜",
    # “唉/哎”类
    "哎": "唉",
    # “嘿嘿/哈哈”简化为单字
    #"嘿嘿": "嘿", "哈哈": "哈",
    # 其它常见
    "噢": "哦", "喔": "哦",
}

# ===== 2) 标点统一：参考原作者并扩展 =====
# 原作者 rep_map（题主给定）为基础：
BASE_REP_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": ",",
    "~": "…",
    "～": "…",
}
# 扩展（自由发挥）：引号/括号去除、连用省略规整、英文分号/冒号等
EXTRA_REP_MAP = {
    "“": "", "”": "", "「": "", "」": "", "『": "", "』": "",
    "（": "", "）": "", "(": "", ")": "",
    "——": ",", "……": "…",
    ";": ",", ":": ",",
}

# 合并映射
REP_MAP = {}
REP_MAP.update(BASE_REP_MAP)
REP_MAP.update(EXTRA_REP_MAP)

# 允许保留的标点集合（统一到半角英文标点）
ALLOWED_PUNCT = set([",", ".", "!", "?", "-", "…"])

def _apply_interj(text: str) -> str:
    t = text
    for src, dst in sorted(INTERJ_MAP.items(), key=lambda x: -len(x[0])):
        t = t.replace(src, dst)
    return t

def _apply_punct_map(text: str) -> str:
    t = text
    for src, dst in sorted(REP_MAP.items(), key=lambda x: -len(x[0])):
        t = t.replace(src, dst)
    return t

def _apply_ellipsis_positional(text: str) -> str:
    """
    将省略号“…”按位置改写：
    - 句末（尾部连续 …）→ “.”
    - 句头（首部连续 …）→ 删除
    - 句中（其余 …）→ “,”
    先处理句末，再处理句头，最后处理中间，避免覆盖冲突。
    """
    t = text
    # 句末 … → .
    t = re.sub(r"…+$", ".", t)
    # 句头 … → 删除
    t = re.sub(r"^…+", "", t)
    # 句中剩余 … → ,
    t = re.sub(r"…+", ",", t)
    return t

def _strip_and_compress(text: str) -> str:
    t = text
    t = re.sub(rf"[^0-9A-Za-z\u4e00-\u9fff{re.escape(''.join(ALLOWED_PUNCT))}]+", "", t)  # 仅保留合法字符
    t = re.sub(r"([,\.\!\?\-…])\1+", r"\1", t)  # 压缩重复标点
    t = t.replace(" ", "")  # 去空格
    t = t.strip(", -")      # 修剪边界逗号/连字符
    if t and t[-1] not in {".", "!", "?", "…"}:  # 句末补句号
        t += "."
    return t

def _normalize_trailing_punct(text: str) -> str:
    """
    清理句末连续标点，只保留优先级最高的一个。
    优先级: , < . < ? < !
    """
    if not text:
        return text
    # 找出句尾连续标点
    m = re.search(r"([,.?!]+)$", text)
    if not m:
        return text
    tail = m.group(1)
    # 选择优先级最高的一个
    priority = {',': 1, '.': 2, '!': 3, '?': 4}
    keep = max(tail, key=lambda ch: priority.get(ch, 0))
    return text[:m.start()] + keep

def _normalize_punct_runs(text: str) -> str:
    """
    全局归一相邻混合标点（不限句尾）：
    - 在任意位置，遇到由 , . ? ! - … 组成的连续串，只保留一个
    - 若串里含终止符(.!?)，按优先级选一个保留
    - 若不含终止符（只有逗号/短横线），统一成一个逗号
    优先级（可改）： , < . < ? < !
    """
    priority = {',': 1, '.': 2, '!': 3, '?': 4}
    # 说明：这里包含了 '-'，因为你常把破折号规整成 '-' 或 ','
    pattern = r"[,\.\!\?\-\u2026]+"
    def pick(m):
        s = m.group(0)
        # 若包含终止符，则按优先级取最高
        terms = [ch for ch in s if ch in ".!?,"]
        if any(ch in ".!?" for ch in terms):
            keep = max(terms, key=lambda ch: priority.get(ch, 0))
            return keep
        # 否则（只有逗号/短横线/省略号残留），统一成逗号
        return ','
    return re.sub(pattern, pick, text)


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", (text or "").strip())
    t = _apply_interj(t)
    t = _apply_punct_map(t)
    # 省略号按位置改写（句末→句号，句头删除，中间→逗号）
    t = _apply_ellipsis_positional(t)
    # [ADD] 全局相邻混合标点归一（不限句尾）
    t = _normalize_punct_runs(t)
    t = _strip_and_compress(t)
    # 句尾连续标点优先级裁剪（保底；可保留，叠加更稳）
    t = _normalize_trailing_punct(t)
    return t

def normalize_csv(in_csv: str, out_csv: str) -> dict:
    df = pd.read_csv(in_csv, encoding="utf-8")
    # 列自适配
    col_file = None
    for c in ["文件名", "语音文件", "新文件名"]:
        if c in df.columns:
            col_file = c
            break
    col_text = "文本" if "文本" in df.columns else None
    if not col_file or not col_text:
        raise ValueError(f"需要列：文件名/语音文件/新文件名 + 文本；当前列：{list(df.columns)}")

    before = df[col_text].astype(str).tolist()
    df[col_text] = df[col_text].astype(str).map(normalize_text)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    return {
        "rows": len(df),
        "sample_before": before[:3],
        "sample_after": df[col_text].head(3).tolist(),
        "out_path": str(Path(out_csv).resolve()),
        "preview": df.head(20),
    }

# ===== 3) 页面：固定绝对路径（可在页面内改），点击“运行一次” =====
def launch_app():
    import gradio as gr  # 仅在启动页面时导入，避免环境不装 gradio 时报错

    DEFAULT_IN = r"D:\lyf\dataset\texts.csv"
    DEFAULT_OUT = r"D:\lyf\dataset\texts_norm.csv"

    with gr.Blocks(title="TTS 文本规范化工具") as demo:
        gr.Markdown("# TTS 文本规范化工具\n- 固定绝对地址（可在下方修改）\n- 点击“运行一次”开始处理")
        in_path = gr.Textbox(label="输入 CSV 绝对路径", value=DEFAULT_IN, lines=1)
        out_path = gr.Textbox(label="输出 CSV 绝对路径", value=DEFAULT_OUT, lines=1)
        run_btn = gr.Button("运行一次", variant="primary")
        status = gr.Markdown()
        preview = gr.Dataframe(interactive=False)
        samples = gr.JSON(label="样例对比（前 3 行）")

        def _on_run(in_csv, out_csv):
            info = normalize_csv(in_csv, out_csv)
            status_md = f"**已完成**：{info['rows']} 行 → 写出到：`{info['out_path']}`"
            return status_md, info["preview"], {"before": info["sample_before"], "after": info["sample_after"]}

        run_btn.click(_on_run, inputs=[in_path, out_path], outputs=[status, preview, samples])

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    # 不走命令行，直接起页面
    launch_app()
