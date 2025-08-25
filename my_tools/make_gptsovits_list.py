# -*- coding: utf-8 -*-
"""
make_list_fixed.py
按固定列名读取CSV（新文件名, 文本），在指定目录按文件名匹配 .wav，
输出 GPT-SoVITS list: vocal_path|speaker|language|text
"""

from pathlib import Path
import csv
import sys

# ===== 配置区（按需修改） =====
CSV_PATH   = Path(r"D:\lyf\GPT-SoVITS-main\datasets\input.csv")  # 你的CSV：必须有表头“文件名,文本”
AUDIO_DIR  = Path(r"D:\lyf\GPT-SoVITS-main\datasets\input")  # 音频根目录（支持子目录）
OUT_LIST   = Path(r"D:\lyf\GPT-SoVITS-main\datasets\input.list")  # 输出list路径

SPEAKER_ID = "spk1"   # 说话人ID（单说话人随便统一写）
LANG_CODE  = "zh"     # 语言：中文=zh（粤语=yue，英文=en 等）
USE_REL_PATH = False  # True=仅写相对于 AUDIO_DIR 的路径；False=写绝对路径
# ===========================


def index_by_filename(root: Path):
    """递归索引 root 下所有 .wav，以纯文件名聚合路径列表。"""
    idx = {}
    for p in root.rglob("*.wav"):
        idx.setdefault(p.name, []).append(p)
    return idx


def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] 找不到CSV：{CSV_PATH}")
        sys.exit(1)
    if not AUDIO_DIR.exists():
        print(f"[ERROR] 找不到音频目录：{AUDIO_DIR}")
        sys.exit(1)

    name_index = index_by_filename(AUDIO_DIR)
    print(f"[INFO] 音频索引完成：{sum(len(v) for v in name_index.values())} 个文件")

    # 读CSV（兼容有/无 BOM）
    try:
        f = open(CSV_PATH, "r", encoding="utf-8-sig", newline="")
        reader = csv.DictReader(f)
    except Exception as e:
        print(f"[ERROR] 打开CSV失败：{e}")
        sys.exit(1)

    # 校验表头
    if not {"文件名", "文本"}.issubset(reader.fieldnames or []):
        print(f"[ERROR] CSV表头必须包含：文件名, 文本；当前={reader.fieldnames}")
        sys.exit(1)

    OUT_LIST.parent.mkdir(parents=True, exist_ok=True)
    out = open(OUT_LIST, "w", encoding="utf-8", newline="")

    n_ok = n_missing = n_multi = 0

    for row in reader:
        fname = (row.get("文件名") or "").strip()
        text  = (row.get("文本") or "").replace("\r", " ").replace("\n", " ").strip()
        if not fname:
            continue
        if not fname.lower().endswith(".wav"):
            fname += ".wav"

        candidates = name_index.get(fname, [])
        if not candidates:
            # 兜底：直接在根目录尝试
            p_try = AUDIO_DIR / fname
            if p_try.exists():
                candidates = [p_try]

        if not candidates:
            n_missing += 1
            print(f"[WARN] 未找到音频：{fname}")
            continue

        if len(candidates) > 1:
            # 多处同名：取路径最短的一个（可按需改）
            candidates = sorted(candidates, key=lambda p: len(p.as_posix()))
            n_multi += 1

        chosen = candidates[0]

        # 写入路径：相对 or 绝对
        if USE_REL_PATH:
            vpath = chosen.relative_to(AUDIO_DIR).as_posix()  # 仅写相对AUDIO_DIR的路径
        else:
            vpath = chosen.as_posix()  # 绝对路径

        # GPT-SoVITS 常用四列
        out.write(f"{vpath}|{SPEAKER_ID}|{LANG_CODE}|{text}\n")
        n_ok += 1

    out.close()
    f.close()

    print("\n[SUMMARY]")
    print(f"  成功写入：{n_ok}")
    print(f"  未找到音频：{n_missing}")
    print(f"  同名多处（已择一）：{n_multi}")
    print(f"[OUT] {OUT_LIST}")


if __name__ == "__main__":
    main()
