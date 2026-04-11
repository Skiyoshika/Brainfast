from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.structure_tree import load_structure_table  # noqa: E402


def test_load_structure_table_supports_official_graph_json(tmp_path):
    payload = {
        "msg": [
            {
                "id": 997,
                "name": "root",
                "acronym": "root",
                "color_hex_triplet": "FFFFFF",
                "children": [
                    {
                        "id": 8,
                        "name": "Basic cell groups and regions",
                        "acronym": "grey",
                        "color_hex_triplet": "BFDAE3",
                        "children": [
                            {
                                "id": 567,
                                "name": "Cerebrum",
                                "acronym": "CH",
                                "color_hex_triplet": "B0F0FF",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    path = tmp_path / "allen_mouse_structure_graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = load_structure_table(path)
    rows = table.set_index("id")

    assert set(table["id"]) == {997, 8, 567}
    assert rows.loc[8, "parent_structure_id"] == 997
    assert rows.loc[567, "parent_structure_id"] == 8
    assert rows.loc[567, "depth"] == 2
    assert rows.loc[567, "structure_id_path"] == "/997/8/567/"


def test_load_structure_table_supports_flat_legacy_json(tmp_path):
    payload = {
        "997": {"name": "root", "acronym": "root", "color": "FFFFFF"},
        "567": {"name": "Cerebrum", "acronym": "CH", "color": "B0F0FF"},
    }
    path = tmp_path / "allen_structure_tree.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = load_structure_table(path)
    rows = table.set_index("id")

    assert rows.loc[997, "structure_id_path"] == "/997/"
    assert rows.loc[567, "acronym"] == "CH"
