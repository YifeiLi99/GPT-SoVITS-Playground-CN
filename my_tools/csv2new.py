import os
import glob
import shutil
import pandas as pd

# ===== 配置 =====
csv_path = "D:\lyf\GPT-SoVITS-main\datasets\paimeng.csv"          # 原始 CSV，至少包含「语音文件」「文本」
src_root = "D:\lyf\GPT-SoVITS-main\datasets\paimeng"             # wav 所在根目录（在此目录下递归查找）
dst_dir  = "D:\lyf\GPT-SoVITS-main\datasets\paimeng_new"     # 新的存放目录（改名后的 wav）
out_csv  = "D:\lyf\GPT-SoVITS-main\datasets\paimeng_new.csv"      # 输出的新 CSV（两列：文件名, 文本）

# ===== 读取 CSV =====
print(f"[INFO] 读取 CSV: {csv_path}")
try:
    df = pd.read_csv(csv_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

for col in ("语音文件", "文本"):
    if col not in df.columns:
        raise ValueError(f"[ERROR] CSV 缺少列: {col}；当前列: {list(df.columns)}")

names = df["语音文件"].astype(str).tolist()
texts = df["文本"].astype(str).tolist()
total = len(names)
print(f"[INFO] 共 {total} 条记录")

# ===== 准备目标文件夹 =====
os.makedirs(dst_dir, exist_ok=True)

def find_in_root(root: str, filename: str):
    """先直接找，再递归搜索一次"""
    direct = os.path.join(root, filename)
    if os.path.exists(direct):
        return direct
    hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
    return hits[0] if hits else None

# ===== 重命名并移动 =====
rows = []   # [文件名, 文本]
miss = 0
for i, (old_name, text) in enumerate(zip(names, texts), start=1):
    src = find_in_root(src_root, old_name)
    # 假设你总 wav 数量 < 100000（五位数），统一补齐到 5 位
    new_name = f"{i:05d}.wav"
    dst_path = os.path.join(dst_dir, new_name)

    if not src:
        print(f"[MISS {i}/{total}] 找不到: {old_name}")
        miss += 1
        continue

    if os.path.exists(dst_path):
        os.remove(dst_path)

    shutil.copy(src, dst_path)   # 建议 copy，如果想剪切改成 shutil.move
    rows.append([new_name, text])
    print(f"[OK   {i}/{total}] {old_name} -> {new_name}")

# ===== 输出新 CSV =====
print(f"[INFO] 写出新 CSV: {out_csv}")
pd.DataFrame(rows, columns=["文件名", "文本"]).to_csv(out_csv, index=False, encoding="utf-8")

print(f"[DONE] 总记录={total} | 成功={len(rows)} | 缺失={miss}")
print(f"[RESULT] 新目录: {os.path.abspath(dst_dir)}")
print(f"[RESULT] 新 CSV: {os.path.abspath(out_csv)}")
