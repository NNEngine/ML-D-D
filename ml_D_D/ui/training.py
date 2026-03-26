"""
ui/training.py
Training state machine: on_run(), on_pause(), on_stop(), per-frame tick,
node highlighting, and CUDA/VRAM menubar stats.
"""

import time
import dearpygui.dearpygui as dpg

import ml_D_D.state as state
from ml_D_D.constants import TRAIN_BTN_STYLES
from ml_D_D.ui.console import log


_cuda_checked = False


def update_cuda_stats() -> None:
    """
    Update CUDA device name and VRAM usage indicators in the menubar.

    This function queries PyTorch for CUDA availability and memory usage,
    then updates UI elements (`mb_cuda`, `mb_vram`) with the current device
    name and VRAM usage. It also color-codes VRAM usage based on utilization.

    Behavior:
        - Shows GPU name when CUDA is available
        - Displays used vs total VRAM (in GB)
        - Applies color coding based on usage ratio
        - Handles cases where CUDA or PyTorch is unavailable

    Returns:
        None
    """
    global _cuda_checked
    if not dpg.does_item_exist("mb_cuda") or not dpg.does_item_exist("mb_vram"):
        return
    try:
        import torch
        if torch.cuda.is_available():
            dev   = torch.cuda.current_device()
            used  = torch.cuda.memory_allocated(dev)  / (1024 ** 3)
            total = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
            ratio = used / total if total > 0 else 0
            vram_col = (
                (80,  220, 120) if ratio < 0.60 else
                (220, 180,  60) if ratio < 0.85 else
                (220,  80,  80)
            )
            dpg.set_value("mb_vram", f"{used:.1f} / {total:.0f} GB")
            dpg.configure_item("mb_vram", color=vram_col)
            if not _cuda_checked:
                dpg.set_value("mb_cuda", torch.cuda.get_device_name(dev))
                dpg.configure_item("mb_cuda", color=(80, 220, 120))
                _cuda_checked = True
        else:
            if not _cuda_checked:
                dpg.set_value("mb_cuda", "Not available")
                dpg.configure_item("mb_cuda", color=(180, 80, 80))
                dpg.set_value("mb_vram", "N/A")
                dpg.configure_item("mb_vram", color=(120, 120, 120))
                _cuda_checked = True
    except ImportError:
        if not _cuda_checked:
            dpg.set_value("mb_cuda", "PyTorch not installed")
            dpg.configure_item("mb_cuda", color=(180, 80, 80))
            dpg.set_value("mb_vram", "N/A")
            dpg.configure_item("mb_vram", color=(120, 120, 120))
            _cuda_checked = True
    except Exception:
        pass


_node_issue_themes: list[int] = []


def highlight_issues(issues: list) -> None:
    """
    Highlight nodes in the UI that have validation issues.

    For each issue, this function:
        - Locates the corresponding node in the active tabs
        - Applies a themed highlight with color-coded severity
        - Stores the applied themes for later cleanup

    Severity mapping:
        - "error"   -> red outline
        - "warning" -> yellow/orange outline

    Args:
        issues (list): List of issue objects containing at least
                       `ntag` (node tag) and `severity` attributes.

    Returns:
        None
    """
    from ml_D_D.engine.blocks import get_block_def
    clear_highlights()
    for issue in issues:
        if not issue.ntag or not dpg.does_item_exist(issue.ntag):
            continue
        block_label = None
        for tab in state.tabs.values():
            if issue.ntag in tab["nodes"]:
                ni = tab["nodes"][issue.ntag]
                block_label = ni["label"] if isinstance(ni, dict) else ni
                break
        block     = get_block_def(block_label) if block_label else None
        title_col = block["color"] if block else (80, 80, 80)
        outline   = (220, 60, 60) if issue.severity == "error" else (220, 180, 40)
        with dpg.theme() as th:
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar,         title_col, category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,  tuple(min(c+30,255) for c in title_col), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, tuple(min(c+50,255) for c in title_col), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline,      outline, category=dpg.mvThemeCat_Nodes)
        dpg.bind_item_theme(issue.ntag, th)
        _node_issue_themes.append(th)


def clear_highlights() -> None:
    """
    Clear all node highlight themes applied due to validation issues.

    This restores each node's original theme (if available) and
    resets the internal theme tracking list.

    Returns:
        None
    """
    global _node_issue_themes
    for tab in state.tabs.values():
        for ntag, ni in tab["nodes"].items():
            if not dpg.does_item_exist(ntag):
                continue
            original = ni["theme"] if isinstance(ni, dict) else 0
            dpg.bind_item_theme(ntag, original if original and dpg.does_item_exist(original) else 0)
    _node_issue_themes = []


def apply_train_btn_style() -> None:
    """
    Apply visual styles and states to training control buttons.

    Updates labels, enabled states, and color themes for the
    Run, Pause, and Stop buttons based on the current training status.

    Styling is derived from the `TRAIN_BTN_STYLES` mapping.

    Returns:
        None
    """
    status = state.train_state["status"]
    styles = TRAIN_BTN_STYLES.get(status, TRAIN_BTN_STYLES["idle"])
    for tag, (label, enabled, col) in zip(["btn_run", "btn_pause", "btn_stop"], styles):
        if not dpg.does_item_exist(tag):
            continue
        dpg.set_item_label(tag, label)
        dpg.configure_item(tag, enabled=enabled)
        with dpg.theme() as th:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (*col, 200))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, tuple(min(c+30,255) for c in col)+(255,))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  tuple(min(c+50,255) for c in col)+(255,))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
        dpg.bind_item_theme(tag, th)


def update_status_indicator() -> None:
    """
    Update the training status indicator in the UI.

    Sets both the color and text of status elements based on the
    current training state (idle, running, paused).

    Returns:
        None
    """
    s    = state.train_state["status"]
    col  = {"idle":(80,220,120),"running":(100,180,255),"paused":(220,180,60)}.get(s,(120,120,120))
    text = {"idle":"Ready","running":"Training","paused":"Paused"}.get(s,"Idle")
    if dpg.does_item_exist("status_dot"):  dpg.configure_item("status_dot", color=col)
    if dpg.does_item_exist("status_text"): dpg.set_value("status_text", text)


def _read_train_config() -> dict:
    """
    Read and construct the training configuration from UI inputs.

    This function safely retrieves configuration values from DearPyGui
    widgets, applies defaults where needed, and resolves checkpoint
    directory paths relative to the current project file.

    Returns:
        dict: Dictionary containing training configuration parameters.
    """
    def _get(tag, fallback):
        return dpg.get_value(tag) if dpg.does_item_exist(tag) else fallback

    import pathlib
    raw_ckpt  = _get("cfg_ckpt_dir", "./checkpoints").strip() or "checkpoints"
    ckpt_path = pathlib.Path(raw_ckpt)
    if not ckpt_path.is_absolute() and state.current_file:
        ckpt_path = pathlib.Path(state.current_file).parent / ckpt_path

    return {
        "epochs":       _get("cfg_epochs",      20),
        "val_split":    _get("cfg_val_split",    0.2),
        "seed":         _get("cfg_seed",         42),
        "shuffle":      _get("cfg_shuffle",      True),
        "device":       _get("cfg_device",       "auto"),
        "amp":          _get("cfg_amp",          False),
        "ckpt_dir":     str(ckpt_path),
        "ckpt_every":   _get("cfg_ckpt_every",   5),
        "ckpt_best":    _get("cfg_ckpt_best",    True),
        "ckpt_monitor": _get("cfg_ckpt_monitor", "val_loss"),
        "es_enable":    _get("cfg_es_enable",    False),
        "es_patience":  _get("cfg_es_patience",  5),
        "es_min_delta": _get("cfg_es_min_delta", 1e-4),
    }


def on_run() -> None:
    """
    Handle the Run button action to start or resume training.

    Behavior:
        - Resumes training if currently paused
        - Validates the pipeline before starting
        - Ensures project is saved and PyTorch is available
        - Highlights validation issues if present
        - Initializes training state and metrics
        - Starts the training process

    Returns:
        None
    """
    from ml_D_D.engine.graph import validate_pipeline
    from ml_D_D.engine.run   import start_training

    ts = state.train_state
    s  = ts["status"]

    if s == "paused":
        from ml_D_D.engine.run import pause_training
        pause_training()
        ts["status"] = "running"
        log("Training resumed.", "info")
        apply_train_btn_style()
        update_status_indicator()
        return

    if s != "idle":
        return

    # Require the project to be saved before training
    if not getattr(state, "current_file", None):
        log("Save your project first before training (Ctrl+S or File > Save As).", "error")
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        log("PyTorch is not installed. Run:  pip install torch torchvision", "error")
        return

    result = validate_pipeline()
    clear_highlights()

    if result.issues:
        highlight_issues(result.issues)
        log("-- Validation --", "header")
        for issue in result.errors:
            log(f"ERROR: {issue.message}" + (f" [{issue.ntag}]" if issue.ntag else ""), "error")
        for issue in result.warnings:
            log(f"WARN:  {issue.message}" + (f" [{issue.ntag}]" if issue.ntag else ""), "warning")

    if not result.ok:
        log(f"Cannot start - {len(result.errors)} error(s). Fix highlighted nodes.", "error")
        apply_train_btn_style()
        update_status_indicator()
        return

    if result.warnings:
        log(f"Validation passed with {len(result.warnings)} warning(s).", "warning")
    else:
        log("Validation passed.", "success")

    cfg   = _read_train_config()
    total = cfg["epochs"]

    log(f"Checkpoints -> {cfg['ckpt_dir']}", "info")

    for key in ("plot_epochs","plot_tl","plot_vl","plot_ta","plot_va","plot_batch_x","plot_batch_y"):
        ts[key] = []

    ts.update({
        "status":             "running",
        "epoch":              0,
        "total_epochs":       total,
        "start_time":         time.time(),
        "real":               True,
        "_last_logged_epoch": 0,
    })
    dpg.set_value("train_progress", 0.0)

    log(f"Starting training - {total} epochs.", "success")

    try:
        from ml_D_D.ui.summary import refresh_model_summary
        refresh_model_summary()
    except Exception:
        pass

    start_training(cfg)
    apply_train_btn_style()
    update_status_indicator()


def on_pause() -> None:
    """
    Toggle the training pause/resume state.

    Behavior:
        - Pauses training if currently running
        - Resumes training if currently paused
        - Updates UI and logs the action

    Returns:
        None
    """
    from ml_D_D.engine.run import pause_training
    ts = state.train_state
    if ts["status"] == "running":
        ts["status"] = "paused"
        pause_training()
        log("Training paused.", "warning")
    elif ts["status"] == "paused":
        ts["status"] = "running"
        pause_training()
        log("Training resumed.", "info")
    apply_train_btn_style()
    update_status_indicator()


def on_stop() -> None:
    """
    Stop the training process and reset training state.

    Behavior:
        - Stops training if running or paused
        - Resets training progress and state variables
        - Clears node highlights
        - Updates UI and logs the action

    Returns:
        None
    """
    from ml_D_D.engine.run import stop_training
    ts = state.train_state
    if ts["status"] in ("running", "paused"):
        stop_training()
        ts.update({"status": "idle", "epoch": 0, "real": False})
        dpg.set_value("train_progress", 0.0)
        log("Training stopped.", "error")
        clear_highlights()
        apply_train_btn_style()
        update_status_indicator()


_cuda_frame_counter  = 0
_split_check_counter = 0


def tick_training(dt: float) -> None:
    """
    Perform per-frame updates related to training.

    This function is called every frame and handles:
        - Periodic CUDA stats updates
        - UI control updates for dataset splitting
        - Processing of training result queue

    Args:
        dt (float): Time delta (seconds) since the last frame.

    Returns:
        None
    """
    global _cuda_frame_counter, _split_check_counter
    _cuda_frame_counter  += 1
    _split_check_counter += 1

    if _cuda_frame_counter >= 60:
        _cuda_frame_counter = 0
        update_cuda_stats()

    if _split_check_counter >= 30:
        _split_check_counter = 0
        _update_split_controls()

    ts = state.train_state
    if ts["status"] not in ("running", "paused"):
        return

    from ml_D_D.engine.run import drain_result_queue
    drain_result_queue()


def _update_split_controls() -> None:
    """
    Enable or disable dataset split controls based on pipeline configuration.

    If a validation DataLoader node exists in the data preparation pipeline,
    manual validation split and shuffle controls are disabled to prevent
    conflicting configurations.

    Returns:
        None
    """
    if not dpg.does_item_exist("cfg_val_split"):
        return
    try:
        from ml_D_D.engine.graph import get_tab_by_role, build_graph
        tab = get_tab_by_role("data_prep")
        has_val_loader = False
        if tab:
            graph = build_graph(tab)
            has_val_loader = any(n.block_label == "DataLoader (val)" for n in graph.values())
        enabled = not has_val_loader
        dpg.configure_item("cfg_val_split", enabled=enabled)
        dpg.configure_item("cfg_shuffle",   enabled=enabled)
    except Exception:
        pass
