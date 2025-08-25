# gradio_asr.py
# ===== 简介 =====
# 一个最小但顺手的 faster-whisper Gradio 界面：
# - 拖拽/上传音频（wav/mp3/m4a 等）
# - 可选语言/任务、VAD、beam size
# - 输出转写文本与语言置信度
# - 端口默认 9874，避免与你 TTS 的 9872 冲突

import gradio as gr
from faster_whisper import WhisperModel
import tempfile
import os
import time
import asr_config
# --------- [ADD][BATCH] 依赖：CSV 读写 ----------
import pandas as pd
import csv


# ===== [ADD] 统一主控台地址（用于单实例切换） =====
CONTROL_URL = "http://127.0.0.1:9900/"

# --------- [ADD] 全局懒加载模型（只加载一次） ----------
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        # 允许传目录或别名 "large-v3"
        model_ref = asr_config.MODEL_DIR if os.path.isdir(getattr(asr_config, "MODEL_DIR", "")) else "large-v3"
        t0 = time.time()
        MODEL = WhisperModel(model_ref, device=getattr(asr_config, "DEVICE", "cuda"),
                             compute_type=getattr(asr_config, "COMPUTE_TYPE", "float16"))
        print(f"[LOAD] model ready in {time.time() - t0:.2f}s -> {model_ref}")
    return MODEL

# --------- [ADD][BATCH] 自动识别音频路径列 ----------
def guess_audio_col(columns):
    """
    从表头里猜测“音频路径”列名。
    优先使用 asr_config.BATCH_AUDIO_COL_CANDIDATES；
    否则按关键词匹配：path/wav/audio/文件/音频/绝对/相对
    """
    # 优先使用配置里的候选
    cand = getattr(asr_config, "BATCH_AUDIO_COL_CANDIDATES", None)
    if cand:
        for c in cand:
            if c in columns:
                return c
    # 关键词兜底（大小写不敏感）
    keys = ["path", "wav", "audio", "文件", "音频", "绝对", "相对"]
    lower_map = {str(c).lower(): c for c in columns}
    for k in keys:
        for lc, orig in lower_map.items():
            if k in lc:
                return orig
    return None

# --------- [ADD] 推理函数 ----------
def transcribe_fn(audio_file, language, task, vad, beam_size):
    """
    audio_file: Gradio 会给你 (sr, np.ndarray) 或文件路径。我们统一成文件路径方便处理。
    language: None / "zh" / "en" ...
    task: "transcribe" or "translate"
    """
    if audio_file is None:
        return "", "未选择音频", ""

    model = get_model()

    # 统一转为临时 wav 文件路径（兼容不同输入形式）
    if isinstance(audio_file, tuple) and len(audio_file) == 2:
        sr, data = audio_file
        import soundfile as sf
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, data, sr)
            wav_path = tmp.name
    elif isinstance(audio_file, str) and os.path.exists(audio_file):
        wav_path = audio_file
    else:
        return "", "无法解析输入的音频格式", ""

    # 推理
    segs, info = model.transcribe(
        wav_path,
        vad_filter=vad,
        language=(None if language in [None, "", "auto"] else language),
        task=task,
        beam_size=int(beam_size),
    )

    text = "".join(s.text for s in segs)
    lang_line = f"lang={info.language} | prob={info.language_probability:.3f}"

    return text, lang_line, os.path.basename(wav_path)

# --------- [ADD][BATCH] 批量 CSV → ASR ----------
def batch_asr_fn(csv_in_path, csv_out_path, language, task, vad, beam):
    """
    读取 csv_in_path，自动找到音频路径列，逐行跑 ASR，
    在末尾新增一列 'asr处理文本'，导出到 csv_out_path（或自动 *_with_asr.csv）。
    返回：日志文本、可下载文件路径（若失败则第二项为空）
    """
    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)

    # 路径检查
    if not csv_in_path or not os.path.exists(csv_in_path):
        return "输入 CSV 不存在，请检查路径。", None

    # 读取 CSV（编码尽量兼容）
    try:
        df = pd.read_csv(csv_in_path, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(csv_in_path, encoding="utf-8-sig")
        except Exception as e:
            return f"读取 CSV 失败：{e}", None

    if df.empty:
        return "CSV 为空，无可处理内容。", None

    # 找到音频列
    audio_col = guess_audio_col(df.columns)
    if audio_col is None:
        return f"无法自动识别音频路径列。请在 CSV 中提供清晰的列名（例如：语音文件/文件路径/wav_path/...）。当前表头：{list(df.columns)}", None

    log(f"[BATCH] 识别到音频列：{audio_col}")

    # 目标输出路径
    if not csv_out_path or str(csv_out_path).strip() == "":
        base, ext = os.path.splitext(csv_in_path)
        csv_out_path = base + "_with_asr" + (ext if ext else ".csv")
        log(f"[BATCH] 未指定输出路径，自动使用：{csv_out_path}")
    else:
        log(f"[BATCH] 指定输出路径：{csv_out_path}")

    # 模型（只加载一次）
    model = get_model()

    # 统一语言/任务/参数
    lang_arg = (None if language in [None, "", "auto"] else language)
    task = task or "transcribe"
    vad = bool(vad)
    beam_size = int(beam) if beam else 1

    # 新列名
    out_col = "asr处理文本"
    if out_col not in df.columns:
        df[out_col] = ""

    ok, fail = 0, 0

    for idx, row in df.iterrows():
        wav_path = str(row[audio_col]).strip()

        if not wav_path or not os.path.exists(wav_path):
            log(f"[SKIP] 第{idx}行：找不到音频文件 -> {wav_path}")
            fail += 1
            continue

        try:
            segs, info = model.transcribe(
                wav_path,
                vad_filter=vad,
                language=lang_arg,
                task=task,
                beam_size=beam_size,
            )
            text = "".join(s.text for s in segs)
            df.at[idx, out_col] = text
            ok += 1
            if (idx + 1) % 10 == 0:
                log(f"[PROG] 已完成 {idx+1}/{len(df)} 行（当前成功 {ok}，失败 {fail}）")
        except Exception as e:
            fail += 1
            log(f"[ERR ] 第{idx}行识别失败：{e}")

    # 导出（全部字段带引号，避免你之前提到的“无引号不安全”问题）
    try:
        df.to_csv(csv_out_path, index=False, encoding="utf-8-sig",
                  quoting=csv.QUOTE_ALL)
        log(f"[DONE] 导出完成：成功 {ok}，失败 {fail}  ->  {csv_out_path}")
        return "\n".join(logs), csv_out_path
    except Exception as e:
        log(f"[FAIL] 写出 CSV 失败：{e}")
        return "\n".join(logs), None

# --------- [ADD] Gradio UI ----------
with gr.Blocks(title="Whisper ASR (faster-whisper)") as demo:
    gr.Markdown("# 🎧 Whisper ASR（faster-whisper）\n上传/拖拽音频即可转写文本。")

    # ===== [ADD] 顶部导航（跳到主控台并自动切到目标模块） =====
    with gr.Row():
        btn_go_tts = gr.Button("TTS", variant="secondary")
        btn_go_eval = gr.Button("结果评测", variant="secondary")
        btn_go_tune = gr.Button("微调", variant="secondary")
        btn_go_clean = gr.Button("数据清洗", variant="secondary")
        btn_go_bridge = gr.Button("语音桥接", variant="secondary")
        btn_go_home = gr.Button("主控台", variant="secondary")
    # —— 绑定：跳到主控台并携带 switch=XXX，让主控台自动关旧开新后再跳转 ——
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
    except TypeError:  # 兼容 Gradio v3.x
        btn_go_tts.click(None, [], [], _js=_js_to_tts)
        btn_go_eval.click(None, [], [], _js=_js_to_eval)
        btn_go_tune.click(None, [], [], _js=_js_to_tune)
        btn_go_clean.click(None, [], [], _js=_js_to_clean)
        btn_go_bridge.click(None, [], [], _js=_js_to_bridge)
        btn_go_home.click(None, [], [], _js=_js_to_home)

    with gr.Row():
        audio = gr.Audio(label="音频文件", sources=["upload", "microphone"], type="filepath")
        with gr.Column():
            language = gr.Dropdown(
                label="语言（留空=自动）",
                choices=["auto", "zh", "en", "ja", "ko", "de", "fr", "es", "ru"],
                value=("auto" if asr_config.UI_DEFAULT_LANGUAGE in [None, "", "auto"] else asr_config.UI_DEFAULT_LANGUAGE),
            )
            task = gr.Radio(
                label="任务",
                choices=["transcribe", "translate"],
                value=asr_config.UI_DEFAULT_TASK
            )
            vad = gr.Checkbox(label="启用 VAD（静音过滤）", value=asr_config.UI_DEFAULT_VAD)
            beam = gr.Slider(label="Beam size", minimum=1, maximum=10, step=1, value=asr_config.UI_DEFAULT_BEAM)
            run_btn = gr.Button("开始识别", variant="primary")

    txt_out = gr.Textbox(label="识别结果", lines=6)
    meta_out = gr.Textbox(label="语言与概率")
    fname_out = gr.Textbox(label="文件名")

    run_btn.click(
        transcribe_fn,
        inputs=[audio, language, task, vad, beam],
        outputs=[txt_out, meta_out, fname_out]
    )

    # --------- [ADD][BATCH] UI：批量 CSV→ASR（默认收起） ----------
    with gr.Accordion("📄 批量CSV→ASR（默认收起）", open=False):
        with gr.Row():
            csv_in_path = gr.Textbox(
                label="CSV 输入路径",
                value=getattr(asr_config, "BATCH_CSV_DEFAULT", ""),
                placeholder=r"D:\path\to\input.csv"
            )
            csv_out_path = gr.Textbox(
                label="输出 CSV 路径（留空=自动 *_with_asr.csv）",
                value=getattr(asr_config, "BATCH_OUTPUT_CSV_DEFAULT", ""),
                placeholder=r"D:\path\to\input_with_asr.csv"
            )
        with gr.Row():
            btn_batch_run = gr.Button("开始批量识别", variant="primary")
        batch_log = gr.Textbox(label="批量日志", lines=10)
        batch_file = gr.File(label="下载输出 CSV")

        # 复用顶部的语言/任务/VAD/beam 控件值，保证与你单条识别一致
        btn_batch_run.click(
            batch_asr_fn,
            inputs=[csv_in_path, csv_out_path, language, task, vad, beam],
            outputs=[batch_log, batch_file]
        )

# --------- [ADD] 启动参数，避免与 9872 冲突 ----------
if __name__ == "__main__":
    # 可按需改 server_name/port（已在 config 里集中）
    demo.queue(api_open=False, max_size=32) if getattr(asr_config, "GRADIO_QUEUE", True) else None
    demo.launch(
        server_name=getattr(asr_config, "GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=getattr(asr_config, "GRADIO_PORT", 9874),
        share=False,              # 关闭外网分享，防止非预期暴露
        inbrowser=False
    )
