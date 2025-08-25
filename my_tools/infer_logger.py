# tools/infer_logger.py
import csv
import time
from pathlib import Path

def _ensure_parent(path: Path):
    """
    工具函数：确保文件所在目录存在。
    如果目录不存在，就自动创建。
    这样在写 CSV 或其他文件时不会因为目录缺失而报错。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

def ensure_csv_header(csv_path: Path):
    """
    功能：确保 CSV 文件存在，并且写入表头（只在文件首次创建时写）。
    - 参数 csv_path: CSV 文件的路径（Path 对象）
    """
    _ensure_parent(csv_path)
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            # 定义日志的表头字段
            w.writerow([
                "timestamp",   # 推理时间点
                "text_raw",    # 原始输入文本
                "text_clean",  # 清洗后的文本
                "token_len",   # token 序列长度
                "mel_shape",   # mel 频谱形状 (80xT 或 NA)
                "audio_sec",   # 音频时长 (秒)
                "infer_sec",   # 推理耗时 (秒)
                "rtf",         # Real-Time Factor = infer_sec / audio_sec
                "sr",          # 采样率
            ])

def wallclock():
    """
    功能：返回当前墙上时钟时间（单位：秒，float）。
    用于计时，配合 dur_sec() 计算耗时。
    """
    return time.time()

def dur_sec(t0, t1):
    """
    功能：计算两个时间戳之间的差值（秒）。
    - 参数 t0: 起始时间
    - 参数 t1: 结束时间
    - 返回值: 耗时秒数 (float)
    """
    return float(max(0.0, t1 - t0))

def try_mel_shape_from_audio(wav, sr, use_librosa=True):
    """
    功能：尝试从最终音频信号后验估计 mel 频谱的形状。
    （非侵入式，不依赖模型内部中间变量）
    - 参数 wav: 音频数组 (numpy.ndarray)，浮点类型
    - 参数 sr: 采样率 (int)
    - 参数 use_librosa: 是否使用 librosa 来计算 mel 频谱
    - 返回值: mel 频谱形状的字符串，如 '80xT'；若失败则返回 'NA'
    """
    if wav is None:
        return "NA"
    if use_librosa:
        try:
            import numpy as np
            import librosa
            # 用 librosa 计算 mel 频谱
            S = librosa.feature.melspectrogram(
                y=wav.astype(float), sr=sr, n_fft=1024, hop_length=256, n_mels=80
            )
            # 返回 "80x帧数"
            return f"{S.shape[0]}x{S.shape[1]}"
        except Exception:
            return "NA"
    return "NA"

def write_csv(csv_path: Path, row: dict):
    """
    功能：往 CSV 文件中追加一行推理日志。
    - 参数 csv_path: CSV 文件路径
    - 参数 row: 字典，包含一次推理的所有指标
    """
    ensure_csv_header(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row.get("timestamp",""),   # 时间戳
            row.get("text_raw",""),    # 原始文本
            row.get("text_clean",""),  # 清洗后文本
            row.get("token_len",""),   # token 序列长度
            row.get("mel_shape",""),   # mel 形状
            row.get("audio_sec",""),   # 音频时长
            row.get("infer_sec",""),   # 推理耗时
            row.get("rtf",""),         # RTF
            row.get("sr",""),          # 采样率
        ])
