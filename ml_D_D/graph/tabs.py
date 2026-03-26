"""
tabs.py

Tab lifecycle management for the graph editor, including creation,
closure, renaming, role assignment, and active-tab synchronization.

This module coordinates UI tab elements with application state and
ensures consistency between DearPyGui and internal tab tracking.
"""

import dearpygui.dearpygui as dpg

import ml_D_D.state as state
from ml_D_D.ui.console import log


# Tag helpers

def editor_tag(tid: int) -> str:
    """
    Generate the node editor tag for a tab.

    Args:
        tid (int): Tab ID.

    Returns:
        str: Editor tag identifier.
    """
    return f"ne_{tid}"


def tab_tag(tid: int) -> str:
    """
    Generate the DearPyGui tab tag.

    Args:
        tid (int): Tab ID.

    Returns:
        str: Tab tag identifier.
    """
    return f"tab_{tid}"


# Accessors

def current_tab() -> dict | None:
    """
    Retrieve the currently active tab state.

    Returns:
        dict | None: Active tab dictionary, or None if no tab is active.
    """
    return state.tabs.get(state.active_tab_id)


# Lifecycle

def new_tab(name: str | None = None, role: str | None = None) -> int:
    """
    Create a new tab and initialize its editor and state.

    This function:
        - Allocates a new tab ID
        - Initializes tab state (nodes, links, undo stacks)
        - Creates a DearPyGui tab and node editor
        - Optionally assigns a pipeline role
        - Sets the tab as active

    Args:
        name (str, optional): Display name of the tab.
        role (str, optional): Pipeline role ("data_prep", "model", "training").

    Returns:
        int: Newly created tab ID.
    """
    from ml_D_D.graph.links import link_callback, delink_callback

    state.tab_counter += 1
    tid  = state.tab_counter
    name = name or f"Graph {tid}"

    state.tabs[tid] = {
        "name":         name,
        "role":         role,
        "editor_tag":   editor_tag(tid),
        "tab_tag":      tab_tag(tid),
        "nodes":        {},
        "links":        {},
        "node_counter": 0,
        "link_counter": 0,
        "undo_stack":   [],
        "redo_stack":   [],
    }

    ttag = tab_tag(tid)
    with dpg.tab(label=name, tag=ttag, parent="canvas_tabbar"):
        cw_tag = f"canvas_cw_{tid}"
        with dpg.child_window(tag=cw_tag, border=False,
                               no_scrollbar=True, width=-1, height=-1):
            with dpg.node_editor(
                tag=editor_tag(tid),
                callback=link_callback,
                delink_callback=delink_callback,
                minimap=True,
                minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
                width=-1,
                height=-1,
            ):
                _add_hint_node(tid, role)

    dpg.set_value("canvas_tabbar", ttag)
    state.active_tab_id = tid
    log(f'Tab "{name}" created.', "debug")
    return tid


_HINT_LINES = {
    "data_prep": [
        "Welcome to the Data Prep tab.",
        "",
        "Build your dataset pipeline here:",
        "  1. Add a Dataset node",
        "     (MNIST, CIFAR10, CIFAR100, FashionMNIST)",
        "  2. Chain transforms    (ToTensor, Normalize, ...)",
        "  3. End with a DataLoader (train) node",
        "  4. Optionally add a second chain ending",
        "     with DataLoader (val) for validation",
        "",
        "Drag nodes from the left palette to get started.",
    ],
    "model": [
        "Welcome to the Model tab.",
        "",
        "Build your neural network here:",
        "  1. Start with an Input node",
        "  2. Add layers  (Linear, Conv2D, ReLU, ...)",
        "  3. End with an Output node",
        "",
        "Connect nodes by dragging from an output pin",
        "to an input pin.",
        "",
        "Drag nodes from the left palette to get started.",
    ],
    "training": [
        "Welcome to the Training tab.",
        "",
        "Wire up your training graph here:",
        "  1. Add ModelBlock and DataLoaderBlock",
        "     (they represent your Model and Data Prep tabs)",
        "  2. Add a Loss node  (CrossEntropyLoss, etc.)",
        "  3. Add an Optimizer  (Adam, SGD, etc.)",
        "  4. Connect:",
        "       DataLoaderBlock.images  -> ModelBlock.images",
        "       ModelBlock.predictions  -> Loss.pred",
        "       DataLoaderBlock.labels  -> Loss.target",
        "       Loss.loss               -> Optimizer.params",
        "",
        "Then press RUN in the toolbar to train.",
    ],
    None: [
        "Drag nodes from the palette on the left.",
        "Connect output pins to input pins to wire them up.",
    ],
}


def _hint_tag(tid: int) -> str:
    """
    Generate the tag for a tab's hint node.

    Args:
        tid (int): Tab ID.

    Returns:
        str: Hint node tag.
    """
    return f"hint_node_{tid}"


def _add_hint_node(tid: int, role) -> None:
    """
    Add an instructional hint node to a tab.

    This node provides guidance based on the tab's role and is displayed
    when the tab is first created.

    Args:
        tid (int): Tab ID.
        role (str | None): Assigned role of the tab.

    Returns:
        None
    """
    lines = _HINT_LINES.get(role, _HINT_LINES[None])
    htag  = _hint_tag(tid)

    with dpg.node(label="Getting Started", tag=htag,
                  pos=(180, 120)):
        with dpg.theme() as hint_theme:
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar,
                                    (50, 50, 50), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,
                                    (60, 60, 60), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected,
                                    (70, 70, 70), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground,
                                    (30, 30, 30, 200), category=dpg.mvThemeCat_Nodes)
        dpg.bind_item_theme(htag, hint_theme)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            for line in lines:
                if line == "":
                    dpg.add_spacer(height=3)
                else:
                    col = (200, 200, 200) if not line.startswith(" ") else (150, 150, 150)
                    dpg.add_text(line, color=col)


def _remove_hint_node(tid: int) -> None:
    """
    Remove the hint node from a tab if it exists.

    Args:
        tid (int): Tab ID.

    Returns:
        None
    """
    htag = _hint_tag(tid)
    if dpg.does_item_exist(htag):
        dpg.delete_item(htag)


def close_tab(tid: int | None) -> None:
    """
    Close a tab and clean up all associated resources.

    This includes deleting nodes, removing UI elements, and updating
    active tab state.

    Args:
        tid (int | None): Tab ID to close.

    Returns:
        None
    """
    from ml_D_D.graph.nodes import raw_delete_node
    from ml_D_D.ui.statusbar import refresh_status

    if tid is None or tid not in state.tabs:
        return

    t = state.tabs[tid]
    for ntag in list(t["nodes"].keys()):
        raw_delete_node(tid, ntag)
    if dpg.does_item_exist(tab_tag(tid)):
        dpg.delete_item(tab_tag(tid))
    del state.tabs[tid]
    log(f'Tab "{t["name"]}" closed.', "debug")

    if state.tabs:
        state.active_tab_id = list(state.tabs.keys())[-1]
        dpg.set_value("canvas_tabbar", tab_tag(state.active_tab_id))
    else:
        state.active_tab_id = None

    refresh_status()


def assign_role(tid: int, role: str | None) -> None:
    """
    Assign or clear a pipeline role for a tab.

    Updates visual styling and triggers pipeline status refresh.

    Args:
        tid (int): Tab ID.
        role (str | None): Role to assign.

    Returns:
        None
    """
    from ml_D_D.graph.pipeline import ROLES, refresh_pipeline_bar
    import ml_D_D.graph.pipeline as _pipeline

    if tid not in state.tabs:
        return

    t = state.tabs[tid]
    t["role"] = role

    ttag = tab_tag(tid)
    if not dpg.does_item_exist(ttag):
        return

    if role and role in ROLES:
        col = ROLES[role]["color"]
        with dpg.theme() as tab_theme:
            with dpg.theme_component(dpg.mvTab):
                dpg.add_theme_color(dpg.mvThemeCol_Text, col)
        dpg.bind_item_theme(ttag, tab_theme)
    else:
        dpg.bind_item_theme(ttag, 0)

    _pipeline._last_pipeline_state = None
    refresh_pipeline_bar()
    log(f'Tab "{t["name"]}" role set to {role or "none"}.', "debug")


def open_assign_role_dialog() -> None:
    """
    Open a dialog for assigning roles to the active tab.

    Returns:
        None
    """
    from ml_D_D.graph.pipeline import ROLES, ROLE_ORDER

    tid = state.active_tab_id
    if tid is None or tid not in state.tabs:
        return

    tag = "assign_role_popup"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    current_role = state.tabs[tid].get("role")

    with dpg.window(label="Assign Tab Role", tag=tag, modal=True,
                    no_resize=True, width=260, height=200,
                    pos=(500, 300)):
        dpg.add_text(f'Assigning role for: {state.tabs[tid]["name"]}',
                     color=(180, 180, 180))
        dpg.add_spacer(height=6)

        for role in ROLE_ORDER:
            info  = ROLES[role]
            label = info["label"]
            col   = info["color"]
            marker = " (current)" if role == current_role else ""

            def _make_cb(r=role, t=tag):
                def cb():
                    assign_role(tid, r)
                    if dpg.does_item_exist(t):
                        dpg.delete_item(t)
                return cb

            dpg.add_button(label=f"{label}{marker}", width=-1,
                           callback=_make_cb())
            with dpg.theme() as btn_th:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button,
                                        (*col, 80))
            dpg.bind_item_theme(dpg.last_item(), btn_th)

        dpg.add_spacer(height=4)
        dpg.add_button(label="Clear role", width=-1,
                       callback=lambda: [
                           assign_role(tid, None),
                           dpg.delete_item(tag) if dpg.does_item_exist(tag) else None,
                       ])
        dpg.add_spacer(height=4)
        dpg.add_button(label="Cancel", width=-1,
                       callback=lambda: dpg.delete_item(tag)
                                        if dpg.does_item_exist(tag) else None)


def rename_tab(tid: int, new_name: str) -> None:
    """
    Rename a tab in both state and UI.

    Args:
        tid (int): Tab ID.
        new_name (str): New tab name.

    Returns:
        None
    """
    if tid not in state.tabs:
        return
    state.tabs[tid]["name"] = new_name
    dpg.set_item_label(tab_tag(tid), new_name)


# Per-frame sync

def sync_active_tab() -> None:
    """
    Synchronize the active tab state with the UI selection.

    This function is called every frame to ensure consistency between
    DearPyGui's tab selection and application state.

    Returns:
        None
    """
    from ml_D_D.ui.statusbar import refresh_status

    if not dpg.does_item_exist("canvas_tabbar"):
        return

    selected_id = dpg.get_value("canvas_tabbar")
    for tid in state.tabs:
        ttag = tab_tag(tid)
        if dpg.does_item_exist(ttag):
            if dpg.get_alias_id(ttag) == selected_id:
                if state.active_tab_id != tid:
                    state.active_tab_id = tid
                    refresh_status()
                break


# Tab change callback

def on_tab_change(sender, app_data) -> None:
    """
    Handle tab selection changes from the UI.

    Args:
        sender: UI element triggering the callback.
        app_data: Selected tab item ID.

    Returns:
        None
    """
    from ml_D_D.ui.statusbar import refresh_status

    for tid in state.tabs:
        ttag = tab_tag(tid)
        if dpg.does_item_exist(ttag) and dpg.get_alias_id(ttag) == app_data:
            state.active_tab_id = tid
            refresh_status()
            break
