"""
state.py
Single source of truth for all shared mutable application state.
"""

import time

tabs:          dict = {}
tab_counter:   int  = 0
active_tab_id: int | None = None

console_lines: list = []

train_state: dict = {
    "status":       "idle",
    "epoch":        0,
    "total_epochs": 20,
    "start_time":   None,
    "real":         False,
}

search_state: dict = {"query": ""}

current_file: str | None = None
