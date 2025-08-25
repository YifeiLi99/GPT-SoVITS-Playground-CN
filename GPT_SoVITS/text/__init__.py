import os
# if os.environ.get("version","v1")=="v1":
#   from text.symbols import symbols
# else:
#   from text.symbols2 import symbols
from my_tools.io_inspector import inspect  # [ADD] 顶部导入一次

from text import symbols as symbols_v1
from text import symbols2 as symbols_v2

_symbol_to_id_v1 = {s: i for i, s in enumerate(symbols_v1.symbols)}
_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2.symbols)}


def cleaned_text_to_sequence(cleaned_text, version=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        phones = [_symbol_to_id_v1[symbol] for symbol in cleaned_text]
    else:
        phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]

    # [ADD] 插桩：检查输入符号和输出 ID 的情况
    inspect(
        "frontend.cleaned_text_to_sequence(符号至ID的映射)",
        input_symbols=list(cleaned_text),  # 输入的符号序列
        ids=phones  # 转换后的 ID 序列
    )

    return phones
