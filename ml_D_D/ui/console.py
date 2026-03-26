"""
ui/console.py

Console logging utilities for rendering timestamped messages in the UI.
This module manages an in-memory log buffer and synchronizes it with the
DearPyGui console panel.
"""

import time
import dearpygui.dearpygui as dpg

import ml_D_D.state as state
from ml_D_D.constants import LOG_COLORS, LOG_PREFIXES, CONSOLE_MAX_LINES


def log(msg: str, level: str = "info") -> None:
    """
    Append a log message to the console buffer and update the UI.

    Each message is timestamped and formatted with a level-specific prefix
    and color. The console maintains a bounded history defined by
    ``CONSOLE_MAX_LINES``.

    Behavior:
        - Prepends current time (HH:MM:SS) to each message
        - Applies color and prefix based on log level
        - Trims oldest entries when exceeding maximum buffer size
        - Triggers a UI refresh after insertion

    Args:
        msg (str): Message text to log.
        level (str, optional): Log severity level (e.g., "info", "success",
            "warning", "error", "header"). Defaults to "info".

    Returns:
        None
    """
    ts     = time.strftime("%H:%M:%S")
    col    = LOG_COLORS.get(level, LOG_COLORS["info"])
    prefix = LOG_PREFIXES.get(level, "     ")
    state.console_lines.append((f"[{ts}] {prefix}  {msg}", col))

    if len(state.console_lines) > CONSOLE_MAX_LINES:
        state.console_lines.pop(0)

    _refresh_console()


def _refresh_console() -> None:
    """
    Re-render the console UI from the current buffer.

    Clears existing UI elements and repopulates them with the current
    console lines, preserving order and color formatting. Automatically
    scrolls to the bottom to show the latest entry.

    Returns:
        None
    """
    if not dpg.does_item_exist("console_content"):
        return
    dpg.delete_item("console_content", children_only=True)
    for line, col in state.console_lines:
        dpg.add_text(line, color=col, parent="console_content")
    dpg.set_y_scroll("console_window", dpg.get_y_scroll_max("console_window"))


def clear_console() -> None:
    """
    Clear all console messages and update the UI.

    Empties the in-memory console buffer and refreshes the display
    to reflect the cleared state.

    Returns:
        None
    """
    state.console_lines.clear()
    _refresh_console()
