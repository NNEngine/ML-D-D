"""
ui/palette.py
Block palette panel: search filtering and button rebuild.
"""

import dearpygui.dearpygui as dpg

import ml_forge.state as state
from ml_forge.engine.blocks import SECTIONS
from ml_forge.constants import SECTION_COLORS


def on_search(sender, app_data) -> None:
    state.search_state["query"] = app_data.lower()
    rebuild_palette()


def rebuild_palette() -> None:
    from ml_forge.graph.nodes import spawn_node

    query = state.search_state["query"]
    dpg.delete_item("palette_content", children_only=True)

    for section_name, categories in SECTIONS.items():
        section_matches = {
            cat: [b for b in blocks if query in b["label"].lower()]
            for cat, blocks in categories.items()
        }
        if not any(section_matches.values()):
            continue

        sec_color  = SECTION_COLORS.get(section_name, (200, 200, 200))
        header_tag = f"section_header_{section_name.replace(' ', '_')}"

        with dpg.collapsing_header(label=section_name, tag=header_tag,
                                   default_open=True, parent="palette_content"):
            with dpg.theme() as hdr_theme:
                with dpg.theme_component(dpg.mvCollapsingHeader):
                    dpg.add_theme_color(dpg.mvThemeCol_Text, sec_color)
            dpg.bind_item_theme(header_tag, hdr_theme)

            for cat_name, blocks in section_matches.items():
                if not blocks:
                    continue
                dpg.add_text(f"  {cat_name}", color=(140, 140, 140), parent=header_tag)
                dpg.add_separator(parent=header_tag)

                for block in blocks:
                    btn_tag = f"palette_btn_{block['label']}"
                    if dpg.does_item_exist(btn_tag):
                        dpg.delete_item(btn_tag)
                    dpg.add_button(label=block["label"], tag=btn_tag, width=150, indent=18,
                                   callback=lambda s, a, u: spawn_node(u),
                                   user_data=block["label"], parent=header_tag)
                    with dpg.theme() as btn_theme:
                        with dpg.theme_component(dpg.mvButton):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, block["color"])
                    dpg.bind_item_theme(btn_tag, btn_theme)

                dpg.add_spacer(height=4, parent=header_tag)