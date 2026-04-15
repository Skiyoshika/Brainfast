from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.asset_bootstrap import (  # noqa: E402
    atlas_asset_status,
    default_structure_source,
    ensure_atlas_assets,
)


def test_default_structure_source_prefers_graph_json_over_flat_tree(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    tree = configs / "allen_structure_tree.json"
    graph = configs / "allen_mouse_structure_graph.json"
    tree.write_text(json.dumps({"997": {"name": "root", "acronym": "root", "color": "FFFFFF"}}))
    graph.write_text(json.dumps({"msg": [{"id": 997, "name": "root", "acronym": "root"}]}))

    assert default_structure_source(tmp_path) == graph


def test_ensure_atlas_assets_uses_existing_local_files_without_network(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (tmp_path / "annotation_25.nii.gz").write_bytes(b"fake-nii")
    graph = configs / "allen_mouse_structure_graph.json"
    graph.write_text(json.dumps({"msg": [{"id": 997, "name": "root", "acronym": "root"}]}))

    status = ensure_atlas_assets(tmp_path, allow_download=False)

    assert status["annotationReady"] is True
    assert status["structureReady"] is True
    assert Path(status["structurePath"]) == graph
    assert atlas_asset_status(tmp_path)["allRequiredReady"] is True
