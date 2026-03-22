"""
filesystem/save.py
Save and load ML Forge project files (.mlf).
"""

from __future__ import annotations

import json
import dearpygui.dearpygui as dpg

import ml_forge.state as state
from ml_forge.ui.console import log

FILE_VERSION = 1


def _serialise_tab(tid: int, t: dict) -> dict:
    nodes_out = []
    for ntag, node_info in t["nodes"].items():
        label  = node_info["label"] if isinstance(node_info, dict) else node_info
        parts  = ntag.split("_")
        nid    = int(parts[2])

        pos = [0, 0]
        if dpg.does_item_exist(ntag):
            raw = dpg.get_item_pos(ntag)
            pos = [int(raw[0]), int(raw[1])]

        params: dict[str, str] = {}
        from ml_forge.engine.blocks import get_block_def
        block = get_block_def(label)
        if block:
            for param in block["params"]:
                ftag = f"node_{parts[1]}_{parts[2]}_input_{param}"
                params[param] = dpg.get_value(ftag) if dpg.does_item_exist(ftag) else ""

        nodes_out.append({"nid": nid, "label": label, "pos": pos, "params": params})

    links_out = []
    for link_tag, (a1, a2) in t["links"].items():
        parts   = link_tag.split("_")
        link_id = int(parts[2])
        if isinstance(a1, int):
            a1 = dpg.get_item_alias(a1) or str(a1)
        if isinstance(a2, int):
            a2 = dpg.get_item_alias(a2) or str(a2)
        links_out.append({"link_id": link_id, "src_attr": a1, "dst_attr": a2})

    return {"name": t["name"], "role": t.get("role"), "nodes": nodes_out, "links": links_out}


def _build_payload() -> dict:
    return {"version": FILE_VERSION,
            "tabs": [_serialise_tab(tid, t) for tid, t in state.tabs.items()]}


def save_project(path: str) -> None:
    try:
        payload = _build_payload()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        state.current_file = path
        log(f"Project saved -> {path}", "success")
        from ml_forge.ui.statusbar import refresh_status
        refresh_status()
    except Exception as e:
        log(f"Save failed: {e}", "error")


def _clear_all_tabs() -> None:
    from ml_forge.graph.nodes import raw_delete_node
    for tid in list(state.tabs.keys()):
        t = state.tabs[tid]
        for ntag in list(t["nodes"].keys()):
            raw_delete_node(tid, ntag)
        ttag = f"tab_{tid}"
        if dpg.does_item_exist(ttag):
            dpg.delete_item(ttag)
    state.tabs.clear()
    state.tab_counter   = 0
    state.active_tab_id = None


def _restore_tab(tab_data: dict) -> None:
    from ml_forge.graph.tabs  import new_tab
    from ml_forge.graph.nodes import raw_spawn_node
    from ml_forge.ui.resize   import resize_callback

    name = tab_data.get("name", "Graph")
    role = tab_data.get("role")

    tid = new_tab(name, role=role)
    dpg.render_dearpygui_frame()
    resize_callback()

    t = state.tabs[tid]

    for node_data in tab_data.get("nodes", []):
        nid    = node_data["nid"]
        label  = node_data["label"]
        pos    = tuple(node_data.get("pos", [0, 0]))
        params = node_data.get("params", {})
        raw_spawn_node(tid, label, nid=nid, pos=pos, params=params)

    # Remove hint node now that real nodes are being loaded
    from ml_forge.graph.tabs import _remove_hint_node
    _remove_hint_node(tid)

    for link_data in tab_data.get("links", []):
        link_id  = link_data["link_id"]
        src_attr = link_data["src_attr"]
        dst_attr = link_data["dst_attr"]
        link_tag = f"link_{tid}_{link_id}"

        if isinstance(src_attr, int):
            src_attr = dpg.get_item_alias(src_attr) or str(src_attr)
        if isinstance(dst_attr, int):
            dst_attr = dpg.get_item_alias(dst_attr) or str(dst_attr)

        if dpg.does_item_exist(src_attr) and dpg.does_item_exist(dst_attr):
            try:
                dpg.add_node_link(src_attr, dst_attr, parent=t["editor_tag"], tag=link_tag)
                t["links"][link_tag] = (src_attr, dst_attr)
                t["link_counter"] = max(t["link_counter"], link_id)
            except Exception as e:
                log(f"Could not restore link {link_tag}: {e}", "warning")
        else:
            log(f"Skipped link {link_tag} - attr not found", "warning")


def load_project(path: str) -> None:
    from ml_forge.ui.statusbar    import refresh_status
    from ml_forge.graph.undo      import refresh_undo_menu
    from ml_forge.graph.pipeline  import refresh_pipeline_bar

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log(f"Load failed - could not read file: {e}", "error")
        return

    version = payload.get("version", 0)
    if version != FILE_VERSION:
        log(f"Warning: file version {version} may not be fully compatible.", "warning")

    _clear_all_tabs()

    tabs_data = payload.get("tabs", [])
    if not tabs_data:
        log("File contains no tabs.", "warning")
        return

    for tab_data in tabs_data:
        _restore_tab(tab_data)

    if state.tabs:
        first_tid = list(state.tabs.keys())[0]
        state.active_tab_id = first_tid
        dpg.set_value("canvas_tabbar", f"tab_{first_tid}")

    state.current_file = path
    refresh_status()
    refresh_undo_menu()
    refresh_pipeline_bar()
    try:
        from ml_forge.engine.autofill import infer_from_dataset
        infer_from_dataset()
    except Exception:
        pass
    log(f"Project loaded <- {path}", "success")


def _make_dialog(label: str, tag: str, callback, default_filename: str = "") -> None:
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    def _on_cancel(s, a):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    with dpg.file_dialog(label=label, tag=tag, callback=callback,
                         cancel_callback=_on_cancel, width=700, height=450,
                         default_filename=default_filename, modal=True):
        dpg.add_file_extension(".mlf",  color=(100, 180, 255), custom_text="ML Forge")
        dpg.add_file_extension(".json", color=(220, 200, 100), custom_text="JSON")
        dpg.add_file_extension(".*")


def open_save_dialog() -> None:
    def _on_save(sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path:
            return
        if not (path.endswith(".mlf") or path.endswith(".json")):
            path += ".mlf"
        save_project(path)
        if dpg.does_item_exist("save_dialog"):
            dpg.delete_item("save_dialog")
    _make_dialog("Save Project", "save_dialog", _on_save, default_filename="project")


def open_load_dialog() -> None:
    def _on_load(sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path:
            return
        load_project(path)
        if dpg.does_item_exist("load_dialog"):
            dpg.delete_item("load_dialog")
    _make_dialog("Open Project", "load_dialog", _on_load)


def save_current() -> None:
    path = getattr(state, "current_file", None)
    if path:
        save_project(path)
    else:
        open_save_dialog()