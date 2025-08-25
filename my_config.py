# my_config.py  [NEW]
from pathlib import Path

# 根目录（保持相对路径）
BASE_DIR = Path(__file__).resolve().parent

# 模型/资源/输出/报告目录（全部相对路径）
MODELS_DIR   = BASE_DIR / "pretrained_models"
OUTPUT_DIR   = BASE_DIR / "outputs"
REPORTS_DIR  = BASE_DIR / "reports"
BATCH_EXPORT_DIR = BASE_DIR / "batch_outputs"

# 日志 CSV（相对路径）
RTF_CSV      = REPORTS_DIR / "rtf_log.csv"

# 固定三句文本
FIXED_SENTENCES = [
    "你好，世界。今天的天气真不错！",
    "请问，现在几点了？我想安排一下行程。",
    "语音合成测试：停顿、韵律与情感，是否能够稳定复现。"
]

# 可选：SR 回落（若推理代码未明确采样率）
DEFAULT_SR = 22050

# 行为开关（不改模型，仅打印/记录）
PRINT_DEBUG = True
WRITE_CSV   = True
SAVE_AUDIO = True  # 需要时改为 False 关闭保存
# [ADD] 打开/关闭节点级插桩的统一开关
INSPECT_IO = True

# 目录兜底创建
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(BASE_DIR)
    print(MODELS_DIR)
    print(OUTPUT_DIR)
    print(REPORTS_DIR)
