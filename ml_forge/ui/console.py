"""
ui/console.py
Console log helper functions.
"""

import time
import dearpygui.dearpygui as dpg

import ml_forge.state as state
from ml_forge.constants import LOG_COLORS, LOG_PREFIXES, CONSOLE_MAX_LINES


def log(msg: str, level: str = "info") -> None:
    ts     = time.strftime("%H:%M:%S")
    col    = LOG_COLORS.get(level, LOG_COLORS["info"])
    prefix = LOG_PREFIXES.get(level, "     ")
    state.console_lines.append((f"[{ts}] {prefix}  {msg}", col))

    if len(state.console_lines) > CONSOLE_MAX_LINES:
        state.console_lines.pop(0)

    _refresh_console()


def _refresh_console() -> None:
    if not dpg.does_item_exist("console_content"):
        return
    dpg.delete_item("console_content", children_only=True)
    for line, col in state.console_lines:
        dpg.add_text(line, color=col, parent="console_content")
    dpg.set_y_scroll("console_window", dpg.get_y_scroll_max("console_window"))


def clear_console() -> None:
    state.console_lines.clear()
    _refresh_console()
