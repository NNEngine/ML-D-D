"""
ui/layout.py
Builds the main DearPyGui window and all child panels:
  - Block palette (left)
  - Canvas tab bar (centre)
  - Training config + plots (right)
  - Console log (bottom)
  - Status bar (bottom edge)
"""

import math
import time
import dearpygui.dearpygui as dpg

from ml_forge.constants      import PALETTE_W, TRAIN_W, CONSOLE_H, STATUSBAR_H
from ml_forge.ui.console     import clear_console
from ml_forge.ui.palette     import on_search
from ml_forge.ui.summary     import refresh_model_summary
from ml_forge.graph.tabs     import on_tab_change

PIPELINE_BAR_H = 24


# Default plot values

_demo_e    = list(range(1, 21))
_demo_tloss = [2.3 * math.exp(-0.18 * e) + 0.05 for e in _demo_e]
_demo_vloss = [2.5 * math.exp(-0.15 * e) + 0.08 for e in _demo_e]
_demo_tacc  = [1 - math.exp(-0.22 * e) * 0.95   for e in _demo_e]
_demo_vacc  = [1 - math.exp(-0.18 * e) * 0.97   for e in _demo_e]


def build_main_window() -> None:
    from ml_forge.ui.toolbar import build_toolbar

    with dpg.window(tag="main_window", no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True, no_collapse=True):
        dpg.add_spacer(height=12)
        build_toolbar()
        dpg.add_separator()
        _build_pipeline_bar()
        dpg.add_separator()
        _build_middle_row()
        dpg.add_separator()
        _build_console()
        dpg.add_separator()
        _build_statusbar()


def _build_toolbar_placeholder() -> None:
    """Toolbar is built separately by ui/toolbar.py — this is a no-op hook."""
    pass  # toolbar.build_toolbar() is called before build_main_window()


def _build_middle_row() -> None:
    with dpg.group(horizontal=True, tag="top_row"):
        _build_palette_panel()
        _build_canvas_panel()
        _build_train_panel()


# Pipeline bar

def _build_pipeline_bar() -> None:
    with dpg.child_window(tag="pipeline_bar", height=PIPELINE_BAR_H,
                           border=False, no_scrollbar=True):
        with dpg.group(horizontal=True, tag="pipeline_bar_content"):
            pass # Pipeline tabs injected by graph.tabs.new_tab() before first render


# Palette

def _build_palette_panel() -> None:
    with dpg.child_window(tag="palette_panel", width=PALETTE_W,
                           border=True, no_scrollbar=False):
        dpg.add_text("Blocks", color=(200, 200, 200))
        dpg.add_separator()
        dpg.add_spacer(height=4)
        dpg.add_input_text(label="", hint="Search...", width=-1,
                           callback=on_search, tag="search_box")
        dpg.add_spacer(height=6)
        with dpg.group(tag="palette_content"):
            pass


# Canvas

def _build_canvas_panel() -> None:
    with dpg.child_window(tag="canvas_panel", border=True, no_scrollbar=True, width=-1, height=-1):
        with dpg.tab_bar(tag="canvas_tabbar",
                         callback=on_tab_change,
                         reorderable=True):
            pass # tabs injected by graph.tabs.new_tab() before first render


# Training config

def _build_train_panel() -> None:
    with dpg.child_window(tag="train_panel", width=TRAIN_W,
                           border=True, no_scrollbar=False):
        dpg.add_text("Training Config", color=(100, 200, 255))
        dpg.add_separator()
        dpg.add_spacer(height=4)

        # Progress bar
        dpg.add_text("Progress", color=(140, 140, 140))
        dpg.add_progress_bar(tag="train_progress", default_value=0.0,
                             width=-1, overlay="Idle")
        dpg.add_spacer(height=6)
        dpg.add_separator()
        dpg.add_spacer(height=4)

        _build_general_section()
        _build_device_section()
        _build_checkpoint_section()
        _build_earlystop_section()
        _build_summary_section()
        _build_loss_plot()
        _build_acc_plot()


def _build_general_section() -> None:
    with dpg.collapsing_header(label="General", default_open=True):
        dpg.add_input_int(label="Epochs",    tag="cfg_epochs",
                          default_value=20,  width=110, min_value=1)
        dpg.add_input_float(label="Val Split", tag="cfg_val_split",
                            default_value=0.2, width=110,
                            format="%.2f", min_value=0.0, max_value=0.99)
        with dpg.tooltip("cfg_val_split"):
            dpg.add_text("Only used when Data Prep has a single DataLoader.\nDisabled when separate DataLoader (val) is present.")
        dpg.add_input_int(label="Seed",      tag="cfg_seed",
                          default_value=42,  width=110)
        dpg.add_checkbox(label="Shuffle",    tag="cfg_shuffle",
                         default_value=True)
        with dpg.tooltip("cfg_shuffle"):
            dpg.add_text("Only used when Data Prep has a single DataLoader.\nDisabled when separate DataLoader (val) is present.")
        dpg.add_spacer(height=4)


def _build_device_section() -> None:
    with dpg.collapsing_header(label="Device", default_open=True):
        dpg.add_combo(label="Device", tag="cfg_device",
                      items=["auto", "cuda", "cpu", "mps"],
                      default_value="auto", width=110)
        dpg.add_checkbox(label="AMP (fp16)", tag="cfg_amp",
                         default_value=False)
        dpg.add_spacer(height=4)


def _build_checkpoint_section() -> None:
    with dpg.collapsing_header(label="Checkpointing", default_open=False):
        dpg.add_input_text(label="Save Dir",    tag="cfg_ckpt_dir",
                           default_value="./checkpoints", width=150)
        dpg.add_input_int(label="Save Every",   tag="cfg_ckpt_every",
                          default_value=5, width=110, min_value=1)
        dpg.add_checkbox(label="Save Best Only", tag="cfg_ckpt_best",
                         default_value=True)
        dpg.add_combo(label="Monitor",          tag="cfg_ckpt_monitor",
                      items=["val_loss", "val_acc", "train_loss"],
                      default_value="val_loss", width=110)
        dpg.add_spacer(height=4)


def _build_earlystop_section() -> None:
    with dpg.collapsing_header(label="Early Stopping", default_open=False):
        dpg.add_checkbox(label="Enable",    tag="cfg_es_enable",
                         default_value=False)
        dpg.add_input_int(label="Patience", tag="cfg_es_patience",
                          default_value=5, width=110, min_value=1)
        dpg.add_input_float(label="Min Delta", tag="cfg_es_min_delta",
                            default_value=1e-4, width=110, format="%.5f")
        dpg.add_spacer(height=4)


def _build_summary_section() -> None:
    with dpg.collapsing_header(label="Model Summary", default_open=True):
        dpg.add_button(label="Refresh", small=True,
                       callback=refresh_model_summary)
        dpg.add_spacer(height=4)
        with dpg.child_window(tag="summary_child", height=220,
                              border=False, no_scrollbar=False):
            with dpg.group(tag="summary_content"):
                dpg.add_text("No nodes on canvas.", color=(120, 120, 120))
        dpg.add_spacer(height=4)


def _build_loss_plot() -> None:
    with dpg.collapsing_header(label="Loss Curve", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_text("Batch smooth:", color=(140, 140, 140))
            dpg.add_slider_int(tag="cfg_batch_smooth", default_value=10,
                               min_value=1, max_value=100, width=-1)
        with dpg.tooltip("cfg_batch_smooth"):
            dpg.add_text("Number of batch loss values to average.\n1 = raw (noisy), higher = smoother line.")
        dpg.add_spacer(height=4)
        with dpg.plot(height=155, width=-1, tag="loss_plot", no_title=True):
            dpg.add_plot_legend()
            xax = dpg.add_plot_axis(dpg.mvXAxis, label="Epoch")
            dpg.set_axis_limits(xax, 0, 22)
            with dpg.plot_axis(dpg.mvYAxis, label="Loss", tag="loss_y"):
                dpg.add_line_series(_demo_e, _demo_tloss,
                                    label="Train", tag="series_train_loss")
                dpg.add_line_series(_demo_e, _demo_vloss,
                                    label="Val",   tag="series_val_loss")
                dpg.add_line_series([], [],
                                    label="Batch", tag="series_batch_loss")
        dpg.add_spacer(height=4)


def _build_acc_plot() -> None:
    with dpg.collapsing_header(label="Accuracy Curve", default_open=True):
        with dpg.plot(height=155, width=-1, tag="acc_plot", no_title=True):
            dpg.add_plot_legend()
            xax = dpg.add_plot_axis(dpg.mvXAxis, label="Epoch")
            dpg.set_axis_limits(xax, 0, 22)
            with dpg.plot_axis(dpg.mvYAxis, label="Accuracy", tag="acc_y"):
                dpg.add_line_series(_demo_e, _demo_tacc,
                                    label="Train", tag="series_train_acc")
                dpg.add_line_series(_demo_e, _demo_vacc,
                                    label="Val",   tag="series_val_acc")
        dpg.add_spacer(height=4)


# Console

def _build_console() -> None:
    with dpg.child_window(tag="console_window", border=True,
                           no_scrollbar=False, horizontal_scrollbar=True):
        with dpg.group(horizontal=True):
            dpg.add_text("Console", color=(100, 200, 255))
            dpg.add_spacer(width=6)
            for lvl, col in [
                ("INFO", (160, 160, 160)),
                ("OK",   (80,  220, 120)),
                ("WARN", (220, 180,  60)),
                ("ERR",  (220,  80,  80)),
            ]:
                dpg.add_text(f"[{lvl}]", color=col)
                dpg.add_spacer(width=4)
            dpg.add_spacer(width=6)
            dpg.add_button(label="Clear", small=True, callback=clear_console)
        dpg.add_separator()
        with dpg.group(tag="console_content"):
            pass


# Status bar

def _build_statusbar() -> None:
    with dpg.child_window(tag="statusbar", height=STATUSBAR_H,
                           border=False, no_scrollbar=True):
        with dpg.group(horizontal=True):
            dpg.add_text("[-]", tag="status_dot",  color=(80, 220, 120))
            dpg.add_text("Ready", tag="status_text", color=(120, 120, 120))
            dpg.add_spacer(width=12)
            dpg.add_text("Nodes: 0   Links: 0",
                         tag="status_nodes", color=(160, 160, 160))
            dpg.add_spacer(width=12)
            dpg.add_text("Undo: 0  Redo: 0",
                         tag="status_undo", color=(120, 120, 140))
            dpg.add_spacer(width=12)
            dpg.add_text("|", color=(60, 60, 60))
            dpg.add_spacer(width=12)
            dpg.add_text("Project:", color=(100, 100, 100))
            dpg.add_input_text(tag="status_project", default_value="untitled",
                               width=140, hint="project name", enabled=False)
            dpg.add_spacer(width=12)
            dpg.add_text("|", color=(60, 60, 60))
            dpg.add_spacer(width=12)
            dpg.add_text(time.strftime("Session: %Y-%m-%d %H:%M"),
                         color=(80, 80, 100))