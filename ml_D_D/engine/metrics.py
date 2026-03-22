"""
metrics.py
Metrics window - shows a live summary of the most recent training run.

Reads directly from state.train_state which is populated by runner.py
during and after training. Can be opened at any time; updates on each
open to reflect the latest data.
"""

from __future__ import annotations
import dearpygui.dearpygui as dpg
import ml_forge.state as state


def open_metrics_window() -> None:
    tag = "metrics_window"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    ww, wh = 540, 580
    px = (vw - ww) // 2
    py = (vh - wh) // 2

    with dpg.window(label="Training Metrics", tag=tag, width=ww, height=wh,
                    pos=(px, py), no_collapse=True, modal=False):

        ts     = state.train_state
        status = ts.get("status", "idle")
        epochs = ts.get("plot_epochs", [])
        tl     = ts.get("plot_tl",     [])
        vl     = ts.get("plot_vl",     [])
        va     = ts.get("plot_va",     [])
        bx     = ts.get("plot_batch_x", [])
        by     = ts.get("plot_batch_y", [])

        has_data = bool(epochs)

        # Header
        status_col = {
            "idle":    (120, 120, 120),
            "running": (100, 180, 255),
            "paused":  (220, 180,  60),
        }.get(status, (120, 120, 120))
        dpg.add_text(f"Status: {status.capitalize()}", color=status_col)
        dpg.add_spacer(height=4)

        if not has_data:
            dpg.add_separator()
            dpg.add_spacer(height=12)
            dpg.add_text("No training data yet.", color=(140, 140, 140))
            dpg.add_text("Run training first, then re-open this window.",
                         color=(100, 100, 100))
            dpg.add_spacer(height=8)
            dpg.add_button(label="Refresh", width=-1,
                           callback=lambda: _refresh(tag))
            return

        # Summary table
        total_epochs = ts.get("total_epochs", len(epochs))
        dpg.add_text(f"Epochs completed: {len(epochs)} / {total_epochs}",
                     color=(200, 200, 200))
        dpg.add_spacer(height=6)
        dpg.add_separator()
        dpg.add_spacer(height=6)

        best_vl_epoch = epochs[vl.index(min(vl))] if vl else None
        best_va_epoch = epochs[va.index(max(va))] if va else None
        final_tl = tl[-1]  if tl else None
        final_vl = vl[-1]  if vl else None
        final_va = va[-1]  if va else None

        def _row(label, value, color=(200, 200, 200)):
            with dpg.group(horizontal=True):
                dpg.add_text(f"{label:<28}", color=(160, 160, 160))
                dpg.add_text(value, color=color)

        dpg.add_text("Final epoch", color=(140, 140, 140))
        dpg.add_spacer(height=2)
        if final_tl is not None:
            _row("  Train loss",       f"{final_tl:.4f}")
        if final_vl is not None:
            _row("  Val loss",         f"{final_vl:.4f}")
        if final_va is not None:
            _row("  Val accuracy",     f"{final_va*100:.2f}%",
                 color=(80, 220, 120) if final_va >= 0.9 else
                       (220, 180, 60) if final_va >= 0.7 else (220, 80, 80))

        dpg.add_spacer(height=8)
        dpg.add_text("Best epoch", color=(140, 140, 140))
        dpg.add_spacer(height=2)
        if vl:
            _row("  Best val loss",    f"{min(vl):.4f}  (epoch {best_vl_epoch})",
                 color=(80, 200, 255))
        if va and max(va) > 0:
            _row("  Best val acc",     f"{max(va)*100:.2f}%  (epoch {best_va_epoch})",
                 color=(80, 220, 120))

        dpg.add_spacer(height=8)
        if tl and vl:
            overfit_gap = final_tl - final_vl
            if overfit_gap < -0.05:
                overfit_msg = "Possible overfitting (val loss > train loss)"
                overfit_col = (220, 180, 60)
            elif abs(overfit_gap) <= 0.05:
                overfit_msg = "Good fit (train and val loss close)"
                overfit_col = (80, 220, 120)
            else:
                overfit_msg = "Underfitting (train loss still high)"
                overfit_col = (220, 130, 60)
            _row("  Fit diagnosis",    overfit_msg, color=overfit_col)

        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=6)

        # Loss curve plot
        dpg.add_text("Loss per epoch", color=(140, 140, 140))
        dpg.add_spacer(height=4)
        with dpg.plot(height=150, width=-1, no_title=True):
            dpg.add_plot_legend()
            xax = dpg.add_plot_axis(dpg.mvXAxis, label="Epoch")
            dpg.set_axis_limits(xax, 1, max(epochs) if epochs else 1)
            with dpg.plot_axis(dpg.mvYAxis, label="Loss"):
                dpg.add_line_series(epochs, tl, label="Train")
                if any(v != t for v, t in zip(vl, tl)):
                    dpg.add_line_series(epochs, vl, label="Val")

        dpg.add_spacer(height=8)

        # Accuracy curve plot
        if va and max(va) > 0:
            dpg.add_text("Accuracy per epoch", color=(140, 140, 140))
            dpg.add_spacer(height=4)
            with dpg.plot(height=150, width=-1, no_title=True):
                dpg.add_plot_legend()
                xax2 = dpg.add_plot_axis(dpg.mvXAxis, label="Epoch")
                dpg.set_axis_limits(xax2, 1, max(epochs) if epochs else 1)
                with dpg.plot_axis(dpg.mvYAxis, label="Accuracy"):
                    dpg.set_axis_limits(dpg.last_item(), 0.0, 1.0)
                    dpg.add_line_series(epochs, va, label="Val acc")
            dpg.add_spacer(height=8)

        # Batch loss plot
        if bx and by:
            dpg.add_text("Batch loss (smoothed)", color=(140, 140, 140))
            dpg.add_spacer(height=4)
            with dpg.plot(height=120, width=-1, no_title=True):
                xax3 = dpg.add_plot_axis(dpg.mvXAxis, label="Epoch (continuous)")
                with dpg.plot_axis(dpg.mvYAxis, label="Loss"):
                    dpg.add_line_series(bx, by, label="Batch loss")
            dpg.add_spacer(height=8)

        # Refresh button
        dpg.add_separator()
        dpg.add_spacer(height=6)
        dpg.add_button(label="Refresh", width=-1,
                       callback=lambda: _refresh(tag))


def _refresh(tag: str) -> None:
    """Close and reopen the window to pull latest data."""
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    open_metrics_window()