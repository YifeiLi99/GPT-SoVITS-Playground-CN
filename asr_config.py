# config.py
MODEL_DIR = "./models/faster-whisper-large-v3"
AUDIO_PATH = "./assets/demo.wav"

BATCH_AUDIO_COL_CANDIDATES = ["文件地址", "语音文件", "音频文件", "wav_path", "audio_path", "path"]
BATCH_CSV_DEFAULT = "./batch_outputs/input.csv"
BATCH_OUTPUT_CSV_DEFAULT = "./batch_outputs/output.csv"

DEVICE = "cuda"      # 或 "cpu"
COMPUTE_TYPE = "float16"   # GPU 可选 "float16"/"int8_float16"，CPU 可选 "int8"

LANGUAGE = "zh"       # 自动检测，中文可写 "zh"
TASK = "transcribe"   # 或 "translate"
VAD = True
BEAM_SIZE = 5

# ===== [ADD] Gradio 相关配置（不与 TTS 的 9872 端口冲突）=====
GRADIO_PORT = 9874          # 避开 9872
GRADIO_SERVER_NAME = "0.0.0.0"  # 需要局域网访问时有用，本机可不改
GRADIO_QUEUE = True         # 开队列防止并发卡顿

# ===== [ADD] ASR 界面默认参数（可在界面里改）=====
UI_DEFAULT_LANGUAGE = "zh"        # None=自动识别；固定中文可改 "zh"
UI_DEFAULT_TASK = "transcribe"    # 或 "translate"
UI_DEFAULT_VAD = True
UI_DEFAULT_BEAM = 5