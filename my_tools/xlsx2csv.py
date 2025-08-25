import pandas as pd

# 输入和输出路径（相对路径，符合你的要求）
input_path = "D:\lyf\GPT-SoVITS-main\datasets\paimeng.xlsx"
output_path = "D:\lyf\GPT-SoVITS-main\datasets\paimeng.csv"

# 读取 Excel 文件
df = pd.read_excel(input_path, dtype=str)  # 以字符串形式读取，避免数值被科学计数法或丢失前导零

# 保存为 UTF-8 编码的 CSV
df.to_csv(output_path, index=False, encoding="utf-8")
