"""
main.py
ML Forge entry point.

Initialises DearPyGui, shows a loading splash, builds the full UI,
creates the three pipeline tabs, and runs the render loop.
"""

import time
import dearpygui.dearpygui as dpg

# Core IMports
from ml_forge.ui.console       import log
from ml_forge.graph.nodes      import delete_selected_nodes
from ml_forge.ui.palette       import rebuild_palette
from ml_forge.graph.pipeline   import refresh_pipeline_bar
from ml_forge.graph.tabs       import new_tab, sync_active_tab
from ml_forge.ui.training      import apply_train_btn_style, tick_training
from ml_forge.graph.undo       import refresh_undo_menu

# UI builders
from ml_forge.ui.menubar import build_menubar
from ml_forge.ui.layout  import build_main_window
from ml_forge.ui.resize  import resize_callback


#  Splash helpers

def _build_splash(vw: int, vh: int) -> None:
    sw, sh = 340, 160
    sx = (vw - sw) // 2
    sy = (vh - sh) // 2

    with dpg.window(tag="splash", no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True, no_collapse=True,
                    modal=False, pos=(sx, sy), width=sw, height=sh):

        dpg.add_spacer(height=18)
        dpg.add_text("ML Forge", tag="splash_title",
                     color=(100, 200, 255))

        # Centre the title text manually after first render
        dpg.add_spacer(height=4)
        dpg.add_text("Initialising...", tag="splash_status",
                     color=(160, 160, 160))
        dpg.add_spacer(height=10)
        dpg.add_progress_bar(tag="splash_progress",
                             default_value=0.0,
                             width=sw - 40,
                             overlay="")
        dpg.add_spacer(height=8)
        dpg.add_text("", tag="splash_step", color=(120, 120, 120))


def _splash_step(label: str, progress: float) -> None:
    if dpg.does_item_exist("splash_status"):
        dpg.set_value("splash_status", label)
    if dpg.does_item_exist("splash_progress"):
        # Animate from current value to target in small increments
        current = dpg.get_value("splash_progress")
        steps   = 8
        for i in range(1, steps + 1):
            v   = current + (progress - current) * (i / steps)
            pct = int(v * 100)
            dpg.set_value("splash_progress", v)
            dpg.configure_item("splash_progress", overlay=f"{pct}%")
            dpg.render_dearpygui_frame()
            time.sleep(0.018)
    else:
        dpg.render_dearpygui_frame()


def _close_splash() -> None:
    if dpg.does_item_exist("splash"):
        dpg.delete_item("splash")


#  Main

def main() -> None:
    dpg.create_context()
    dpg.create_viewport(title="ML Forge", width=1380, height=820,
                        resizable=True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    _build_splash(vw, vh)
    dpg.render_dearpygui_frame()

    _splash_step("Building UI...", 0.10)
    build_menubar()
    build_main_window()
    dpg.set_viewport_resize_callback(resize_callback)
    dpg.set_primary_window("main_window", True)
    resize_callback()

    import ml_forge.state as state
    from ml_forge.graph.tabs import tab_tag

    for tab_name, tab_role in [
        ("Data Prep", "data_prep"),
        ("Model",     "model"),
        ("Training",  "training"),
    ]:
        new_tab(tab_name, role=tab_role)

    if dpg.does_item_exist("canvas_tab_dummy"):
        dpg.delete_item("canvas_tab_dummy")

    first_tid = list(state.tabs.keys())[0]
    state.active_tab_id = first_tid
    dpg.set_value("canvas_tabbar", tab_tag(first_tid))

    _splash_step("UI built.", 0.30)

    _splash_step("Loading block palette...", 0.40)
    rebuild_palette()
    _splash_step("Palette loaded.", 0.55)

    _splash_step("Tabs ready.", 0.78)

    _splash_step("Finalising...", 0.82)
    apply_train_btn_style()
    refresh_undo_menu()
    refresh_pipeline_bar()
    from ml_forge.ui.training import update_cuda_stats
    update_cuda_stats()

    _splash_step("Ready.", 1.0)
    time.sleep(0.3)

    _close_splash()

    log("ML Forge ready.", "header")
    log("Build your pipeline across the three tabs: Data Prep -> Model -> Training.", "info")
    log("Press RUN when all three pipeline stages are complete.", "info")

    prev_time       = time.time()
    _autofill_counter = 0

    while dpg.is_dearpygui_running():
        now       = time.time()
        dt        = now - prev_time
        prev_time = now
        
        if dpg.is_key_pressed(dpg.mvKey_Delete):
            delete_selected_nodes()

        if dpg.is_key_down(dpg.mvKey_LControl):
            if dpg.is_key_pressed(dpg.mvKey_S):
                from ml_forge.filesystem.save import save_current
                save_current()
            if dpg.is_key_pressed(dpg.mvKey_Z):
                from ml_forge.graph.undo import undo
                undo()
            if dpg.is_key_pressed(dpg.mvKey_Y):
                from ml_forge.graph.undo import redo
                redo()

        sync_active_tab()
        refresh_pipeline_bar()
        tick_training(dt)

        _autofill_counter += 1
        if _autofill_counter >= 30:
            _autofill_counter = 0
            try:
                t = state.tabs.get(state.active_tab_id)
                if t and t.get("role") == "model":
                    from ml_forge.engine.autofill import on_param_changed
                    on_param_changed(t)
            except Exception:
                pass

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()