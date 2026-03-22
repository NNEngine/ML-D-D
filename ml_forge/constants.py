"""
constants.py
All numeric layout constants and tuneable behaviour values.
"""

PALETTE_W    = 210
TRAIN_W      = 265
CONSOLE_H    = 150
TOOLBAR_H    = 28
TABBAR_H     = 26
STATUSBAR_H  = 22
MENUBAR_H    = 40

MAX_UNDO = 40

NODE_GRID_COLS    = 5
NODE_GRID_X_STEP  = 220
NODE_GRID_Y_STEP  = 180
NODE_GRID_ORIGIN  = (40, 40)

CONSOLE_MAX_LINES = 300

SECTION_COLORS = {
    "Model Creation": (100, 180, 255),
    "Training":       (100, 220, 180),
    "Data Prep":      (220, 180, 255),
}

LOG_COLORS = {
    "info":    (200, 200, 200),
    "success": (80,  220, 120),
    "warning": (220, 180,  60),
    "error":   (220,  80,  80),
    "debug":   (140, 140, 180),
    "header":  (100, 200, 255),
}

LOG_PREFIXES = {
    "info":    "INFO ",
    "success": "OK   ",
    "warning": "WARN ",
    "error":   "ERR  ",
    "debug":   "DBG  ",
    "header":  "     ",
}

TRAIN_BTN_STYLES = {
    "idle":    [
        ("RUN",   True,  (60,  140,  60)),
        ("PAUSE", False, (60,   60,  60)),
        ("STOP",  False, (60,   60,  60)),
    ],
    "running": [
        ("RUN",   False, (40,   80,  40)),
        ("PAUSE", True,  (140, 120,  40)),
        ("STOP",  True,  (140,  50,  50)),
    ],
    "paused":  [
        ("RUN",   True,  (60,  140,  60)),
        ("PAUSE", True,  (160, 140,  50)),
        ("STOP",  True,  (140,  50,  50)),
    ],
}
