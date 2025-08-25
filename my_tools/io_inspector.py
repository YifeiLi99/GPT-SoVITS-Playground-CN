import numpy as np
import torch
from collections.abc import Mapping, Sequence

# [ADD] 模块级说明：
# 本文件提供“节点级插桩”能力：在关键函数尾部调用 inspect(tag, **kv)
# 将输入/输出的 shape、统计量（mean/std/min/max）打印到控制台。
# 仅打印，不修改任何数据或返回值；可通过 config.INSPECT_IO 总开关静音。

# ---------------------------------------------------------------------------
# 内部工具函数（下划线前缀表示仅供本模块内部调用）
# ---------------------------------------------------------------------------

def _il_num_stats(x):
    """
    [作用] 对 numpy 数组或可转为 ndarray 的对象做摘要统计。
    [返回] 'shape=(...), mean=..., std=..., min=..., max=..., dtype=...' 的字符串。
    [异常] 任意异常都会被捕获并返回 '<num_stats_error:...>'，保证不影响主流程。
    """
    try:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.size == 0:
            return "shape=%s | empty" % (tuple(x.shape),)
        return "shape=%s | mean=%.4f std=%.4f min=%.4f max=%.4f dtype=%s" % (
            tuple(x.shape), x.mean(), x.std(), x.min(), x.max(), x.dtype
        )
    except Exception as e:
        return f"<num_stats_error:{e}>"

def _il_torch_stats(t):
    """
    [作用] 对 torch.Tensor 做摘要统计；GPU 张量会临时转为 CPU float 计算。
    [返回] 同 _il_num_stats 的统计字符串，并保留原 dtype。
    """
    try:
        if not isinstance(t, torch.Tensor):
            return _il_num_stats(t)
        if t.numel() == 0:
            return f"shape={tuple(t.shape)} | empty | dtype={t.dtype}"
        x = t.detach()
        x = (x.float().cpu() if x.is_cuda else x.float())
        return "shape=%s | mean=%.4f std=%.4f min=%.4f max=%.4f dtype=%s" % (
            tuple(t.shape),
            x.mean().item(),
            x.std(unbiased=False).item(),
            x.min().item(),
            x.max().item(),
            t.dtype,
        )
    except Exception as e:
        return f"<torch_stats_error:{e}>"

def _il_summarize(k, v, max_items=6, level=0):
    """
    [作用] 统一的对象摘要：Tensor/ndarray 输出统计；dict/list 递归展示前若干项；
          字符串只展示前 60 字符。用于构造最终打印内容。
    """
    pad = "  " * level
    try:
        if isinstance(v, torch.Tensor):
            return f"{pad}{k}: {_il_torch_stats(v)}"
    except Exception:
        pass
    try:
        if isinstance(v, np.ndarray):
            return f"{pad}{k}: {_il_num_stats(v)}"
    except Exception:
        pass
    if isinstance(v, str):
        short = v[:60] + ("…" if len(v) > 60 else "")
        return f'{pad}{k}: str(len={len(v)}): "{short}"'
    if isinstance(v, Mapping):
        lines = [f"{pad}{k}: dict(len={len(v)})"]
        for i, (kk, vv) in enumerate(v.items()):
            if i >= max_items:
                lines.append(f"{pad}  …(+{len(v)-max_items} more)")
                break
            lines.append(_il_summarize(f"[{kk}]", vv, max_items, level+1))
        return "\n".join(lines)
    if isinstance(v, Sequence) and not isinstance(v, (bytes, bytearray, str)):
        lines = [f"{pad}{k}: list(len={len(v)})"]
        for i, vv in enumerate(v[:max_items]):
            lines.append(_il_summarize(f"[{i}]", vv, max_items, level+1))
        if len(v) > max_items:
            lines.append(f"{pad}  …(+{len(v)-max_items} more)")
        return "\n".join(lines)
    return f"{pad}{k}: {type(v).__name__}({str(v)[:60]}{'…' if len(str(v))>60 else ''})"

# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------

def inspect(tag: str, **kwargs):
    """
    节点级插桩：只打印，不改行为。
    用法: inspect("frontend.clean", inp=text, out=cleaned)
    可通过 config.INSPECT_IO 开关一键关闭。
    """
    try:
        from config import INSPECT_IO
    except Exception:
        INSPECT_IO = True
    if not INSPECT_IO:
        return

    lines = []
    sep = "─" * 80  # 分隔线（80 个 ─）
    lines.append(sep)
    lines.append(f"[INSPECT] {tag}")
    for k, v in kwargs.items():
        lines.append(_il_summarize(k, v))
    lines.append(sep)
    print("\n".join(lines))
