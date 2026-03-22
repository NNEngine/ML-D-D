"""
ui/menubar.py
Builds the DearPyGui viewport menu bar.
All callbacks are imported from their respective logic modules.
"""

import dearpygui.dearpygui as dpg
import ml_forge.state as state


def build_menubar() -> None:
    from ml_forge.graph.nodes       import delete_selected_nodes, clear_canvas
    from ml_forge.graph.tabs        import new_tab, close_tab, open_assign_role_dialog
    from ml_forge.graph.undo        import undo, redo
    from ml_forge.ui.training       import on_run, on_pause, on_stop
    from ml_forge.engine.generator  import export_pytorch
    from ml_forge.filesystem.save   import save_current, open_save_dialog, open_load_dialog

    with dpg.viewport_menu_bar():
        dpg.add_text("ML Forge", color=(100, 200, 255))
        dpg.add_text("v1.0",     color=(120, 120, 140))
        dpg.add_separator()

        # File
        with dpg.menu(label="File"):
            dpg.add_separator()
            dpg.add_menu_item(label="Open...",    callback=open_load_dialog)
            dpg.add_menu_item(label="Save",       callback=save_current)
            dpg.add_menu_item(label="Save As...", callback=open_save_dialog)
            dpg.add_separator()
            with dpg.menu(label="Templates"):
                dpg.add_menu_item(label="MNIST Classifier",
                                  callback=lambda: _load_template("mnist_classifier.mlf"))
                dpg.add_menu_item(label="CIFAR10 Classifier",
                                  callback=lambda: _load_template("cifar10_classifier.mlf"))
            dpg.add_separator()
            with dpg.menu(label="Export"):
                with dpg.menu(label="Python"):
                    dpg.add_menu_item(label="PyTorch",    callback=export_pytorch)

        # Edit
        with dpg.menu(label="Edit"):
            dpg.add_menu_item(label="Undo", tag="menu_undo",
                              callback=undo, enabled=False)
            dpg.add_menu_item(label="Redo", tag="menu_redo",
                              callback=redo, enabled=False)
            dpg.add_separator()
            dpg.add_menu_item(label="Delete Selected  [Del]",
                              callback=delete_selected_nodes)
            dpg.add_separator()
            dpg.add_menu_item(label="Clear Canvas", callback=clear_canvas)

        # Graph
        with dpg.menu(label="Graph"):
            dpg.add_menu_item(label="New Tab",
                              callback=lambda: new_tab())
            dpg.add_menu_item(label="Close Tab",
                              callback=lambda: close_tab(state.active_tab_id))
            dpg.add_separator()
            dpg.add_menu_item(label="Assign Role...",
                              callback=open_assign_role_dialog)

        # Run
        with dpg.menu(label="Run"):
            dpg.add_menu_item(label="Inference...",
                              callback=lambda: __import__("ml_forge.engine.inference", fromlist=["open_inference_window"]).open_inference_window())

        # Help
        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="Documentation", callback=_open_docs)
            dpg.add_menu_item(label="About ML Forge", callback=_open_about)

        dpg.add_separator()
        dpg.add_spacer(width=6)

        # Run controls
        dpg.add_button(label="RUN",     tag="btn_run",   small=True, callback=on_run)
        dpg.add_button(label="PAUSE",   tag="btn_pause", small=True,
                       callback=on_pause, enabled=False)
        dpg.add_button(label="STOP",    tag="btn_stop",  small=True,
                       callback=on_stop, enabled=False)
        
        dpg.add_separator()
        dpg.add_button(label="METRICS", tag="btn_metrics", small=True,
                       callback=lambda: __import__("ml_forge.engine.metrics", fromlist=["open_metrics_window"]).open_metrics_window())

        dpg.add_separator()
        dpg.add_text("VRAM:", color=(150, 150, 150))
        dpg.add_text("...", tag="mb_vram", color=(220, 180, 60))
        dpg.add_separator()
        dpg.add_text("CUDA:", color=(150, 150, 150))
        dpg.add_text("...", tag="mb_cuda", color=(120, 120, 120))


def _load_template(filename: str) -> None:
    """Load a bundled template without overwriting state.current_file."""
    import pathlib
    from ml_forge.filesystem.save import load_project
    from ml_forge.ui.console import log
    import ml_forge.state as state

    templates_dir = pathlib.Path(__file__).parent.parent / "templates"
    path = templates_dir / filename

    if not path.exists():
        log(f"Template not found: {path}", "error")
        return

    load_project(str(path))
    state.current_file = None
    log("Template loaded. Use File > Save As to save your own copy.", "info")
    from ml_forge.ui.statusbar import refresh_status
    refresh_status()


def _open_docs() -> None:
    tag = "docs_window"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    ww, wh = 560, 520
    with dpg.window(label="Documentation", tag=tag,
                    width=ww, height=wh,
                    pos=((vw-ww)//2, (vh-wh)//2),
                    no_collapse=True):

        dpg.add_text("ML Forge - Quick Reference", color=(100, 200, 255))
        dpg.add_separator()
        dpg.add_spacer(height=6)

        with dpg.collapsing_header(label="Getting Started", default_open=True):
            dpg.add_text("1. Data Prep tab - build your dataset and transform chain")
            dpg.add_text("   Add a Dataset node (MNIST, CIFAR10, etc.)")
            dpg.add_text("   Chain transforms: ToTensor, Normalize, augmentations")
            dpg.add_text("   End with DataLoader (train) and optionally DataLoader (val)")
            dpg.add_spacer(height=4)
            dpg.add_text("2. Model tab - build your neural network")
            dpg.add_text("   Start with Input, end with Output")
            dpg.add_text("   Chain layers: Linear, Conv2D, ReLU, BatchNorm2D, etc.")
            dpg.add_spacer(height=4)
            dpg.add_text("3. Training tab - configure training")
            dpg.add_text("   Connect DataLoaderBlock -> ModelBlock -> Loss -> Optimizer")
            dpg.add_text("   Configure epochs, device, checkpointing in the right panel")
            dpg.add_text("   Press RUN in the menubar to start training")

        dpg.add_spacer(height=4)
        with dpg.collapsing_header(label="Keyboard Shortcuts", default_open=True):
            dpg.add_text("Del          Delete selected nodes")
            dpg.add_text("Ctrl+Drag    Pan the canvas")
            dpg.add_text("Middle-drag  Pan the canvas (built-in)")
            dpg.add_text("Ctrl+S       Save project")
            dpg.add_text("Ctrl+Z       Undo")
            dpg.add_text("Ctrl+Y       Redo")

        dpg.add_spacer(height=4)
        with dpg.collapsing_header(label="Training Tab Wiring", default_open=True):
            dpg.add_text("Add these nodes from the palette (Training section):")
            dpg.add_text("  ModelBlock, DataLoaderBlock, a Loss, and an Optimizer")
            dpg.add_spacer(height=4)
            dpg.add_text("Required connections:")
            dpg.add_text("  DataLoaderBlock.images  -> ModelBlock.images")
            dpg.add_text("  ModelBlock.predictions  -> Loss.pred")
            dpg.add_text("  DataLoaderBlock.labels  -> Loss.target")
            dpg.add_text("  Loss.loss               -> Optimizer.params")
            dpg.add_spacer(height=4)
            dpg.add_text("RUN will fail validation if any of these are missing.")

        dpg.add_spacer(height=4)
        with dpg.collapsing_header(label="Datasets", default_open=False):
            dpg.add_text("Supported: MNIST, CIFAR10, CIFAR100, FashionMNIST, ImageFolder")
            dpg.add_text("Set root to a local folder. Enable download=True on first run.")
            dpg.add_text("Use two Dataset nodes (train=True and train=False) with")
            dpg.add_text("DataLoader (train) and DataLoader (val) for proper splits.")

        dpg.add_spacer(height=4)
        with dpg.collapsing_header(label="Checkpoints", default_open=False):
            dpg.add_text("Checkpoints are saved to the Save Dir in Training Config.")
            dpg.add_text("best.pth  - best validation checkpoint")
            dpg.add_text("final.pth - checkpoint at end of training")
            dpg.add_text("Use File > Run > Inference to test a saved checkpoint.")

        dpg.add_spacer(height=8)
        dpg.add_button(label="Close", width=-1,
                       callback=lambda: dpg.delete_item(tag) if dpg.does_item_exist(tag) else None)


def _open_about() -> None:
    tag = "about_window"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    ww, wh = 380, 280
    with dpg.window(label="About ML Forge", tag=tag,
                    width=ww, height=wh,
                    pos=((vw-ww)//2, (vh-wh)//2),
                    no_collapse=True, no_resize=True):

        dpg.add_spacer(height=8)
        dpg.add_text("ML Forge", color=(100, 200, 255))
        dpg.add_text("Version 1.0", color=(160, 160, 160))
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=8)
        dpg.add_text("A visual PyTorch pipeline editor.")
        dpg.add_text("Build, train and run ML models without writing code.")
        dpg.add_spacer(height=8)
        dpg.add_text("Built with:", color=(140, 140, 140))
        dpg.add_text("  DearPyGui  - UI framework")
        dpg.add_text("  PyTorch    - Training engine")
        dpg.add_text("  torchvision - Datasets and transforms")
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=4)
        dpg.add_text("Templates included:", color=(140, 140, 140))
        dpg.add_text("  MNIST Classifier")
        dpg.add_text("  CIFAR10 Classifier")
        dpg.add_spacer(height=8)
        dpg.add_button(label="Close", width=-1,
                       callback=lambda: dpg.delete_item(tag) if dpg.does_item_exist(tag) else None)