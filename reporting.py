"""统一的终端表格输出。"""

from collections.abc import Iterable, Sequence
import unicodedata


def _display_width(text: str) -> int:
    """计算终端显示宽度；中日韩全角字符通常占两个英文字符位。"""
    return sum(
        2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
        for character in text
    )


def _pad_right(text: str, width: int) -> str:
    return text + " " * (width - _display_width(text))


def format_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """把数据格式化成便于终端观察的 Markdown 风格表格。"""
    header_cells = [str(cell) for cell in headers]
    body = [[str(cell) for cell in row] for row in rows]
    if not header_cells:
        raise ValueError("表头不能为空")
    if any(len(row) != len(header_cells) for row in body):
        raise ValueError("每一行的列数必须与表头一致")

    widths = [_display_width(cell) for cell in header_cells]
    for row in body:
        widths = [
            max(width, _display_width(cell))
            for width, cell in zip(widths, row, strict=True)
        ]

    def render(row: Sequence[str]) -> str:
        cells = [
            f" {_pad_right(cell, width)} "
            for cell, width in zip(row, widths, strict=True)
        ]
        return "|" + "|".join(cells) + "|"

    separator = "|" + "|".join(f" {'-' * width} " for width in widths) + "|"
    lines = [render(header_cells), separator]
    lines.extend(render(row) for row in body)
    return "\n".join(lines)


def format_key_values(rows: Iterable[tuple[object, object]]) -> str:
    """两列表格快捷入口，适合配置、shape 和环境信息。"""
    return format_table(("项目", "值"), rows)
