import os
import sys

os.environ["version"] = version = "v2ProPlus"
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import warnings

warnings.filterwarnings("ignore")
import json
import platform
import shutil
import signal

import psutil
import torch
import yaml

# [ADD] 主控台 URL（用于跨模块切换）
CONTROL_URL = os.environ.get("CONTROL_URL", "http://127.0.0.1:9900/")

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if os.path.exists(tmp):
    for name in os.listdir(tmp):
        if name == "jieba.cache":
            continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site
import traceback

site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if site_packages_roots == []:
    site_packages_roots = ["%s/runtime/Lib/site-packages" % now_dir]
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    # "%s\n%s/runtime\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    "%s\n%s/GPT_SoVITS/BigVGAN\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            traceback.print_exc()
import shutil
import subprocess
from subprocess import Popen

from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"] = language
i18n = I18nAuto(language=language)
from multiprocessing import cpu_count

from config import (
    GPU_INDEX,
    GPU_INFOS,
    IS_GPU,
    exp_root,
    infer_device,
    is_half,
    is_share,
    memset,
    python_exec,
    webui_port_infer_tts,
    webui_port_main,
)
from tools import my_utils
from tools.my_utils import check_details, check_for_existance

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr

n_cpu = cpu_count()

set_gpu_numbers = GPU_INDEX
gpu_infos = GPU_INFOS
mem = memset
is_gpu_ok = IS_GPU

v3v4set = {"v3", "v4"}


def set_default():
    global \
        default_batch_size, \
        default_max_batch_size, \
        gpu_info, \
        default_sovits_epoch, \
        default_sovits_save_every_epoch, \
        max_sovits_epoch, \
        max_sovits_save_every_epoch, \
        default_batch_size_s1, \
        if_force_ckpt
    if_force_ckpt = False
    gpu_info = "\n".join(gpu_infos)
    if is_gpu_ok:
        minmem = min(mem)
        default_batch_size = minmem // 2 if version not in v3v4set else minmem // 8
        default_batch_size_s1 = minmem // 2
    else:
        default_batch_size = default_batch_size_s1 = int(psutil.virtual_memory().total / 1024 / 1024 / 1024 / 4)
    if version not in v3v4set:
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25  # 40
        max_sovits_save_every_epoch = 25  # 10
    else:
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 16  # 40 # 3 #训太多=作死
        max_sovits_save_every_epoch = 10  # 10 # 3

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3


set_default()

gpus = "-".join(map(str, GPU_INDEX))
default_gpu_numbers = infer_device.index


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


from config import pretrained_gpt_name, pretrained_sovits_name


def check_pretrained_is_exist(version):
    pretrained_model_list = (
        pretrained_sovits_name[version],
        pretrained_sovits_name[version].replace("s2G", "s2D"),
        pretrained_gpt_name[version],
        "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    )
    _ = ""
    for i in pretrained_model_list:
        if "s2Dv3" not in i and "s2Dv4" not in i and os.path.exists(i) == False:
            _ += f"\n    {i}"
    if _:
        print("warning: ", i18n("以下模型不存在:") + _)


check_pretrained_is_exist(version)
for key in pretrained_sovits_name.keys():
    if os.path.exists(pretrained_sovits_name[key]) == False:
        pretrained_sovits_name[key] = ""
for key in pretrained_gpt_name.keys():
    if os.path.exists(pretrained_gpt_name[key]) == False:
        pretrained_gpt_name[key] = ""

from config import (
    GPT_weight_root,
    GPT_weight_version2root,
    SoVITS_weight_root,
    SoVITS_weight_version2root,
    change_choices,
    get_weights_names,
)

for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)
SoVITS_names, GPT_names = get_weights_names()


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()


def kill_process(pid, process_name=""):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        kill_proc_tree(pid)
    print(process_name + i18n("进程已终止"))


def process_info(process_name="", indicator=""):
    if indicator == "opened":
        return process_name + i18n("已开启")
    elif indicator == "open":
        return i18n("开启") + process_name
    elif indicator == "closed":
        return process_name + i18n("已关闭")
    elif indicator == "close":
        return i18n("关闭") + process_name
    elif indicator == "running":
        return process_name + i18n("运行中")
    elif indicator == "occupy":
        return process_name + i18n("占用中") + "," + i18n("需先终止才能开启下一次任务")
    elif indicator == "finish":
        return process_name + i18n("已完成")
    elif indicator == "failed":
        return process_name + i18n("失败")
    elif indicator == "info":
        return process_name + i18n("进程输出信息")
    else:
        return process_name


p_train_SoVITS = None
process_name_sovits = i18n("SoVITS训练")


def open1Ba(
        version,
        batch_size,
        total_epoch,
        exp_name,
        text_low_lr_rate,
        if_save_latest,
        if_save_every_weights,
        save_every_epoch,
        gpu_numbers1Ba,
        pretrained_s2G,
        pretrained_s2D,
        if_grad_ckpt,
        lora_rank,
):
    global p_train_SoVITS
    if p_train_SoVITS == None:
        exp_name = exp_name.rstrip(" ")
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        with open(config_file) as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2_%s" % (s2_dir, version), exist_ok=True)
        if check_for_existance([s2_dir], is_train=True):
            check_details([s2_dir], is_train=True)
        if is_half == False:
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["train"]["grad_ckpt"] = if_grad_ckpt
        data["train"]["lora_rank"] = lora_rank
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_version2root[version]
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))
        if version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
            cmd = '"%s" -s GPT_SoVITS/s2_train.py --config "%s"' % (python_exec, tmp_config_path)
        else:
            cmd = '"%s" -s GPT_SoVITS/s2_train_v3_lora.py --config "%s"' % (python_exec, tmp_config_path)
        yield (
            process_info(process_name_sovits, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        yield (
            process_info(process_name_sovits, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            process_info(process_name_sovits, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1Ba():
    global p_train_SoVITS
    if p_train_SoVITS is not None:
        kill_process(p_train_SoVITS.pid, process_name_sovits)
        p_train_SoVITS = None
    return (
        process_info(process_name_sovits, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


p_train_GPT = None
process_name_gpt = i18n("GPT训练")


def open1Bb(
        batch_size,
        total_epoch,
        exp_name,
        if_dpo,
        if_save_latest,
        if_save_every_weights,
        save_every_epoch,
        gpu_numbers,
        pretrained_s1,
):
    global p_train_GPT
    if p_train_GPT == None:
        exp_name = exp_name.rstrip(" ")
        with open(
                "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml"
        ) as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        if check_for_existance([s1_dir], is_train=True):
            check_details([s1_dir], is_train=True)
        if is_half == False:
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_version2root[version]
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)

        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(gpu_numbers.replace("-", ","))
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" -s GPT_SoVITS/s1_train.py --config_file "%s" ' % (python_exec, tmp_config_path)
        yield (
            process_info(process_name_gpt, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        yield (
            process_info(process_name_gpt, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            process_info(process_name_gpt, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1Bb():
    global p_train_GPT
    if p_train_GPT is not None:
        kill_process(p_train_GPT.pid, process_name_gpt)
        p_train_GPT = None
    return (
        process_info(process_name_gpt, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1a = []
process_name_1a = i18n("文本分词与特征提取")


def open1a(inp_text, inp_wav_dir, exp_name, gpu_numbers, bert_pretrained_dir):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1a == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "bert_pretrained_dir": bert_pretrained_dir,
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    "is_half": str(is_half),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1a.append(p)
        yield (
            process_info(process_name_1a, "running"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a = []
        if len("".join(opt)) > 0:
            yield (
                process_info(process_name_1a, "finish"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
        else:
            yield (
                process_info(process_name_1a, "failed"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
    else:
        yield (
            process_info(process_name_1a, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1a():
    global ps1a
    if ps1a != []:
        for p1a in ps1a:
            try:
                kill_process(p1a.pid, process_name_1a)
            except:
                traceback.print_exc()
        ps1a = []
    return (
        process_info(process_name_1a, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
ps1b = []
process_name_1b = i18n("语音自监督特征提取")


def open1b(version, inp_text, inp_wav_dir, exp_name, gpu_numbers, ssl_pretrained_dir):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1b == []:
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": "%s/%s" % (exp_root, exp_name),
            "cnhubert_base_dir": ssl_pretrained_dir,
            "sv_path": sv_path,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)
        yield (
            process_info(process_name_1b, "running"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        for p in ps1b:
            p.wait()
        ps1b = []
        if "Pro" in version:
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-sv.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1b.append(p)
            for p in ps1b:
                p.wait()
            ps1b = []
        yield (
            process_info(process_name_1b, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            process_info(process_name_1b, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1b():
    global ps1b
    if ps1b != []:
        for p1b in ps1b:
            try:
                kill_process(p1b.pid, process_name_1b)
            except:
                traceback.print_exc()
        ps1b = []
    return (
        process_info(process_name_1b, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1c = []
process_name_1c = i18n("语义Token提取")


def open1c(version, inp_text, inp_wav_dir, exp_name, gpu_numbers, pretrained_s2G_path):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1c == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        config = {
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": config_file,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1c.append(p)
        yield (
            process_info(process_name_1c, "running"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c = []
        yield (
            process_info(process_name_1c, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            process_info(process_name_1c, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1c():
    global ps1c
    if ps1c != []:
        for p1c in ps1c:
            try:
                kill_process(p1c.pid, process_name_1c)
            except:
                traceback.print_exc()
        ps1c = []
    return (
        process_info(process_name_1c, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1abc = []
process_name_1abc = i18n("训练集格式化一键三连")


def open1abc(
        version,
        inp_text,
        inp_wav_dir,
        exp_name,
        gpu_numbers1a,
        gpu_numbers1Ba,
        gpu_numbers1c,
        bert_pretrained_dir,
        ssl_pretrained_dir,
        pretrained_s2G_path,
):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1abc == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        try:
            #############################1a
            path_text = "%s/2-name2text.txt" % opt_dir
            if os.path.exists(path_text) == False or (
                    os.path.exists(path_text) == True
                    and len(open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2
            ):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half),
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    i18n("进度") + ": 1A-Doing",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
                for p in ps1abc:
                    p.wait()

                opt = []
                for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, process_info(process_name_1a, "failed")
            yield (
                i18n("进度") + ": 1A-Done",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            ps1abc = []
            #############################1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
                "sv_path": sv_path,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield (
                i18n("进度") + ": 1A-Done, 1B-Doing",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            for p in ps1abc:
                p.wait()
            ps1abc = []
            if "Pro" in version:
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-sv.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                for p in ps1abc:
                    p.wait()
                ps1abc = []
            yield (
                i18n("进度") + ": 1A-Done, 1B-Done",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if os.path.exists(path_semantic) == False or (
                    os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31
            ):
                config_file = (
                    "GPT_SoVITS/configs/s2.json"
                    if version not in {"v2Pro", "v2ProPlus"}
                    else f"GPT_SoVITS/configs/s2{version}.json"
                )
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": config_file,
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    i18n("进度") + ": 1A-Done, 1B-Done, 1C-Doing",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
                for p in ps1abc:
                    p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield (
                    i18n("进度") + ": 1A-Done, 1B-Done, 1C-Done",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
            ps1abc = []
            yield (
                process_info(process_name_1abc, "finish"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
        except:
            traceback.print_exc()
            close1abc()
            yield (
                process_info(process_name_1abc, "failed"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
    else:
        yield (
            process_info(process_name_1abc, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1abc():
    global ps1abc
    if ps1abc != []:
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid, process_name_1abc)
            except:
                traceback.print_exc()
        ps1abc = []
    return (
        process_info(process_name_1abc, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


def switch_version(version_):
    os.environ["version"] = version_
    global version
    version = version_
    if pretrained_sovits_name[version] != "" and pretrained_gpt_name[version] != "":
        ...
    else:
        gr.Warning(i18n("未下载模型") + ": " + version.upper())
    set_default()
    return (
        {"__type__": "update", "value": pretrained_sovits_name[version]},
        {"__type__": "update", "value": pretrained_sovits_name[version].replace("s2G", "s2D")},
        {"__type__": "update", "value": pretrained_gpt_name[version]},
        {"__type__": "update", "value": default_batch_size, "maximum": default_max_batch_size},
        {"__type__": "update", "value": default_sovits_epoch, "maximum": max_sovits_epoch},
        {"__type__": "update", "value": default_sovits_save_every_epoch, "maximum": max_sovits_save_every_epoch},
        {"__type__": "update", "visible": True if version not in v3v4set else False},
        {
            "__type__": "update",
            "value": False if not if_force_ckpt else True,
            "interactive": True if not if_force_ckpt else False,
        },
        {"__type__": "update", "visible": True if version in v3v4set else False},
    )


if os.path.exists("GPT_SoVITS/text/G2PWModel"):
    ...
else:
    cmd = '"%s" -s GPT_SoVITS/download.py' % python_exec
    p = Popen(cmd, shell=True)
    p.wait()


def sync(text):
    return {"__type__": "update", "value": text}


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:
    gr.HTML(
        top_html.format(
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.")
            + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        ),
        elem_classes="markdown",
    )

    # [ADD] 顶部导航：跳到主控台并交给主控台切换模块
    with gr.Row():
        btn_go_tts = gr.Button("TTS", variant="secondary")
        btn_go_asr = gr.Button("ASR", variant="secondary")
        btn_go_eval = gr.Button("结果评测", variant="secondary")
        btn_go_clean = gr.Button("数据清洗", variant="secondary")
        btn_go_bridge = gr.Button("语音桥接", variant="secondary")
        btn_go_home = gr.Button("主控台", variant="secondary")
    # [ADD] 绑定：跳到主控台并携带 switch=XXX，让主控台执行“关旧→开新”后再跳转
    _js_to_tts = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=TTS'; return []; }}"
    _js_to_asr = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=ASR'; return []; }}"
    _js_to_eval = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=EVAL'; return []; }}"
    _js_to_clean = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=CLEAN'; return []; }}"
    _js_to_bridge = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}?switch=BRIDGE'; return []; }}"
    _js_to_home = f"(x)=>{{ window.top.location.href = '{CONTROL_URL}'; return []; }}"
    try:  # Gradio v4+
        btn_go_tts.click(None, [], [], js=_js_to_tts)
        btn_go_asr.click(None, [], [], js=_js_to_asr)
        btn_go_eval.click(None, [], [], js=_js_to_eval)
        btn_go_clean.click(None, [], [], js=_js_to_clean)
        btn_go_bridge.click(None, [], [], js=_js_to_bridge)
        btn_go_home.click(None, [], [], js=_js_to_home)
    except TypeError:  # 兼容 Gradio v3.x
        btn_go_tts.click(None, [], [], _js=_js_to_tts)
        btn_go_asr.click(None, [], [], _js=_js_to_asr)
        btn_go_eval.click(None, [], [], _js=_js_to_eval)
        btn_go_clean.click(None, [], [], _js=_js_to_clean)
        btn_go_bridge.click(None, [], [], _js=_js_to_bridge)
        btn_go_home.click(None, [], [], _js=_js_to_home)

    with gr.Tabs():
        with gr.Accordion(i18n("微调模型信息")):
            with gr.Row():
                with gr.Row(equal_height=True):
                    exp_name = gr.Textbox(
                        label=i18n("*实验/模型名"),
                        value="xxx",
                        interactive=True,
                        scale=3,
                    )
                    gpu_info_box = gr.Textbox(
                        label=i18n("显卡信息"),
                        value=gpu_info,
                        visible=True,
                        interactive=False,
                        scale=5,
                    )
                    version_checkbox = gr.Radio(
                        label=i18n("训练模型的版本"),
                        value=version,
                        choices=["v1", "v2", "v4", "v2Pro", "v2ProPlus"],
                        scale=5,
                    )

        with gr.Accordion(label=i18n("预训练模型路径")):
            with gr.Row():
                with gr.Row(equal_height=True):
                    pretrained_s1 = gr.Textbox(
                        label=i18n("预训练GPT模型路径"),
                        value=pretrained_gpt_name[version],
                        interactive=True,
                        lines=1,
                        max_lines=1,
                        scale=3,
                    )
                    pretrained_s2G = gr.Textbox(
                        label=i18n("预训练SoVITS-G模型路径"),
                        value=pretrained_sovits_name[version],
                        interactive=True,
                        lines=1,
                        max_lines=1,
                        scale=5,
                    )
                    pretrained_s2D = gr.Textbox(
                        label=i18n("预训练SoVITS-D模型路径"),
                        value=pretrained_sovits_name[version].replace("s2G", "s2D"),
                        interactive=True,
                        lines=1,
                        max_lines=1,
                        scale=5,
                    )

        with gr.TabItem("1A-" + i18n("训练集格式化工具")):
            with gr.Accordion(label=i18n("请填写list文件所在的绝对地址")):
                with gr.Row():
                    with gr.Row():
                        inp_text = gr.Textbox(
                            label=i18n("*文本标注文件"),
                            value=r"D:\RVC1006\GPT-SoVITS\raw\xxx.list",
                            interactive=True,
                            scale=10,
                        )
                    with gr.Row():
                        inp_wav_dir = gr.Textbox(
                            label=i18n("*训练集音频文件目录"),
                            # value=r"D:\RVC1006\GPT-SoVITS\raw\xxx",
                            interactive=True,
                            placeholder=i18n(
                                "留空即可，留空会使用.list文件里的绝对路径"
                            ),
                            scale=10,
                        )

            with gr.Accordion(label="1Aa-" + process_name_1a, open=False):
                with gr.Row():
                    with gr.Row():
                        gpu_numbers1a = gr.Textbox(
                            label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                        )
                    with gr.Row():
                        bert_pretrained_dir = gr.Textbox(
                            label=i18n("预训练中文BERT模型路径"),
                            value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                            interactive=False,
                            lines=2,
                        )
                    with gr.Row():
                        button1a_open = gr.Button(
                            value=process_info(process_name_1a, "open"), variant="primary", visible=True
                        )
                        button1a_close = gr.Button(
                            value=process_info(process_name_1a, "close"), variant="primary", visible=False
                        )
                    with gr.Row():
                        info1a = gr.Textbox(label=process_info(process_name_1a, "info"))

            with gr.Accordion(label="1Ab-" + process_name_1b, open=False):
                with gr.Row():
                    with gr.Row():
                        gpu_numbers1Ba = gr.Textbox(
                            label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                        )
                    with gr.Row():
                        cnhubert_base_dir = gr.Textbox(
                            label=i18n("预训练SSL模型路径"),
                            value="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                            interactive=False,
                            lines=2,
                        )
                    with gr.Row():
                        button1b_open = gr.Button(
                            value=process_info(process_name_1b, "open"), variant="primary", visible=True
                        )
                        button1b_close = gr.Button(
                            value=process_info(process_name_1b, "close"), variant="primary", visible=False
                        )
                    with gr.Row():
                        info1b = gr.Textbox(label=process_info(process_name_1b, "info"))

            with gr.Accordion(label="1Ac-" + process_name_1c, open=False):
                with gr.Row():
                    with gr.Row():
                        gpu_numbers1c = gr.Textbox(
                            label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                        )
                    with gr.Row():
                        pretrained_s2G_ = gr.Textbox(
                            label=i18n("预训练SoVITS-G模型路径"),
                            value=pretrained_sovits_name[version],
                            interactive=False,
                            lines=2,
                        )
                    with gr.Row():
                        button1c_open = gr.Button(
                            value=process_info(process_name_1c, "open"), variant="primary", visible=True
                        )
                        button1c_close = gr.Button(
                            value=process_info(process_name_1c, "close"), variant="primary", visible=False
                        )
                    with gr.Row():
                        info1c = gr.Textbox(label=process_info(process_name_1c, "info"))

            with gr.Accordion(label="1Aabc-" + process_name_1abc):
                with gr.Row():
                    with gr.Row():
                        button1abc_open = gr.Button(
                            value=process_info(process_name_1abc, "open"), variant="primary", visible=True
                        )
                        button1abc_close = gr.Button(
                            value=process_info(process_name_1abc, "close"), variant="primary", visible=False
                        )
                    with gr.Row():
                        info1abc = gr.Textbox(label=process_info(process_name_1abc, "info"))

        pretrained_s2G.change(sync, [pretrained_s2G], [pretrained_s2G_])

        button1a_open.click(
            open1a,
            [inp_text, inp_wav_dir, exp_name, gpu_numbers1a, bert_pretrained_dir],
            [info1a, button1a_open, button1a_close],
        )
        button1a_close.click(close1a, [], [info1a, button1a_open, button1a_close])
        button1b_open.click(
            open1b,
            [version_checkbox, inp_text, inp_wav_dir, exp_name, gpu_numbers1Ba, cnhubert_base_dir],
            [info1b, button1b_open, button1b_close],
        )
        button1b_close.click(close1b, [], [info1b, button1b_open, button1b_close])
        button1c_open.click(
            open1c,
            [version_checkbox, inp_text, inp_wav_dir, exp_name, gpu_numbers1c, pretrained_s2G],
            [info1c, button1c_open, button1c_close],
        )
        button1c_close.click(close1c, [], [info1c, button1c_open, button1c_close])
        button1abc_open.click(
            open1abc,
            [
                version_checkbox,
                inp_text,
                inp_wav_dir,
                exp_name,
                gpu_numbers1a,
                gpu_numbers1Ba,
                gpu_numbers1c,
                bert_pretrained_dir,
                cnhubert_base_dir,
                pretrained_s2G,
            ],
            [info1abc, button1abc_open, button1abc_close],
        )
        button1abc_close.click(close1abc, [], [info1abc, button1abc_open, button1abc_close])

        with gr.TabItem("1B-" + i18n("微调训练")):
            with gr.Accordion(label="1Ba-" + i18n("SoVITS 训练: 模型权重文件在 SoVITS_weights/")):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=default_max_batch_size,
                                step=1,
                                label=i18n("每张显卡的batch_size"),
                                value=default_batch_size,
                                interactive=True,
                            )
                            total_epoch = gr.Slider(
                                minimum=1,
                                maximum=max_sovits_epoch,
                                step=1,
                                label=i18n("总训练轮数total_epoch，不建议太高"),
                                value=default_sovits_epoch,
                                interactive=True,
                            )
                        with gr.Row():
                            text_low_lr_rate = gr.Slider(
                                minimum=0.2,
                                maximum=0.6,
                                step=0.05,
                                label=i18n("文本模块学习率权重"),
                                value=0.4,
                                visible=True if version not in v3v4set else False,
                            )  # v3v4 not need
                            lora_rank = gr.Radio(
                                label=i18n("LoRA秩"),
                                value="32",
                                choices=["16", "32", "64", "128"],
                                visible=True if version in v3v4set else False,
                            )  # v1v2 not need
                            save_every_epoch = gr.Slider(
                                minimum=1,
                                maximum=max_sovits_save_every_epoch,
                                step=1,
                                label=i18n("保存频率save_every_epoch"),
                                value=default_sovits_save_every_epoch,
                                interactive=True,
                            )
                    with gr.Column():
                        with gr.Column():
                            if_save_latest = gr.Checkbox(
                                label=i18n("是否仅保存最新的权重文件以节省硬盘空间"),
                                value=True,
                                interactive=True,
                                show_label=True,
                            )
                            if_save_every_weights = gr.Checkbox(
                                label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                                value=True,
                                interactive=True,
                                show_label=True,
                            )
                            if_grad_ckpt = gr.Checkbox(
                                label="v3是否开启梯度检查点节省显存占用",
                                value=False,
                                interactive=True if version in v3v4set else False,
                                show_label=True,
                                visible=False,
                            )  # 只有V3s2可以用
                        with gr.Row():
                            gpu_numbers1Ba = gr.Textbox(
                                label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                value="%s" % (gpus),
                                interactive=True,
                            )
                with gr.Row():
                    with gr.Row():
                        button1Ba_open = gr.Button(
                            value=process_info(process_name_sovits, "open"), variant="primary", visible=True
                        )
                        button1Ba_close = gr.Button(
                            value=process_info(process_name_sovits, "close"), variant="primary", visible=False
                        )
                    with gr.Row():
                        info1Ba = gr.Textbox(label=process_info(process_name_sovits, "info"))
            with gr.Accordion(label="1Bb-" + i18n("GPT 训练: 模型权重文件在 GPT_weights/")):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            batch_size1Bb = gr.Slider(
                                minimum=1,
                                maximum=40,
                                step=1,
                                label=i18n("每张显卡的batch_size"),
                                value=default_batch_size_s1,
                                interactive=True,
                            )
                            total_epoch1Bb = gr.Slider(
                                minimum=2,
                                maximum=50,
                                step=1,
                                label=i18n("总训练轮数total_epoch"),
                                value=15,
                                interactive=True,
                            )
                        with gr.Row():
                            save_every_epoch1Bb = gr.Slider(
                                minimum=1,
                                maximum=50,
                                step=1,
                                label=i18n("保存频率save_every_epoch"),
                                value=5,
                                interactive=True,
                            )
                            if_dpo = gr.Checkbox(
                                label=i18n("是否开启DPO训练选项(实验性)"),
                                value=False,
                                interactive=True,
                                show_label=True,
                            )
                    with gr.Column():
                        with gr.Column():
                            if_save_latest1Bb = gr.Checkbox(
                                label=i18n("是否仅保存最新的权重文件以节省硬盘空间"),
                                value=True,
                                interactive=True,
                                show_label=True,
                            )
                            if_save_every_weights1Bb = gr.Checkbox(
                                label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                                value=True,
                                interactive=True,
                                show_label=True,
                            )
                        with gr.Row():
                            gpu_numbers1Bb = gr.Textbox(
                                label=i18n("GPU卡号以-分割，每个卡号一个进程"),
                                value="%s" % (gpus),
                                interactive=True,
                            )
                with gr.Row():
                    with gr.Row():
                        button1Bb_open = gr.Button(
                            value=process_info(process_name_gpt, "open"), variant="primary", visible=True
                        )
                        button1Bb_close = gr.Button(
                            value=process_info(process_name_gpt, "close"), variant="primary", visible=False
                        )
                    with gr.Row():
                        info1Bb = gr.Textbox(label=process_info(process_name_gpt, "info"))

        button1Ba_close.click(close1Ba, [], [info1Ba, button1Ba_open, button1Ba_close])
        button1Bb_close.click(close1Bb, [], [info1Bb, button1Bb_open, button1Bb_close])

        button1Ba_open.click(
            open1Ba,
            [
                version_checkbox,
                batch_size,
                total_epoch,
                exp_name,
                text_low_lr_rate,
                if_save_latest,
                if_save_every_weights,
                save_every_epoch,
                gpu_numbers1Ba,
                pretrained_s2G,
                pretrained_s2D,
                if_grad_ckpt,
                lora_rank,
            ],
            [info1Ba, button1Ba_open, button1Ba_close],
        )
        button1Bb_open.click(
            open1Bb,
            [
                batch_size1Bb,
                total_epoch1Bb,
                exp_name,
                if_dpo,
                if_save_latest1Bb,
                if_save_every_weights1Bb,
                save_every_epoch1Bb,
                gpu_numbers1Bb,
                pretrained_s1,
            ],
            [info1Bb, button1Bb_open, button1Bb_close],
        )
        version_checkbox.change(
            switch_version,
            [version_checkbox],
            [
                pretrained_s2G,
                pretrained_s2D,
                pretrained_s1,
                batch_size,
                total_epoch,
                save_every_epoch,
                text_low_lr_rate,
                if_grad_ckpt,
                lora_rank,
            ],
        )

    # [FIX] 仅使用 FINETUNE 专用变量或默认 9882，避免被全局 WEBUI_PORT 影响
    DEFAULT_FINETUNE_PORT = 9882
    _fin_port = int(os.environ.get("WEBUI_PORT_FINETUNE", str(DEFAULT_FINETUNE_PORT)))
    print(f"[FINETUNE] Using port: {_fin_port}")

    app.queue().launch(  # concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=False,
        share=is_share,
        server_port=_fin_port,
    )
