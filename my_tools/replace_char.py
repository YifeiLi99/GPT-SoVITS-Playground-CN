import pandas as pd

def replace_char_in_csv(input_csv, output_csv, old_char, new_char):
    # 读取 CSV
    df = pd.read_csv(input_csv, encoding="utf-8")

    # 遍历每一列，逐列替换
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(old_char, new_char, regex=False)

    # 保存到新文件
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"替换完成：'{old_char}' → '{new_char}'，结果已保存到 {output_csv}")


if __name__ == "__main__":
    # 修改这里参数
    input_csv = "D:\lyf\GPT-SoVITS-main\datasets\paimeng.csv"      # 输入文件
    output_csv = "D:\lyf\GPT-SoVITS-main\datasets\paimeng1.csv"    # 输出文件
    old_char = "诶"              # 要替换的字符
    new_char = "欸"              # 替换后的字符

    replace_char_in_csv(input_csv, output_csv, old_char, new_char)
