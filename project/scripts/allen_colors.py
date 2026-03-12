from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_allen_color_map(structure_csv: Path) -> dict[int, tuple[int, int, int]]:
    """Load Allen-like color mapping from CSV if rgb columns exist.
    Falls back to deterministic palette elsewhere in renderer.
    """
    if not structure_csv.exists():
        return {}
    df = pd.read_csv(structure_csv)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get('id') or cols.get('region_id')
    r_col = cols.get('r') or cols.get('red')
    g_col = cols.get('g') or cols.get('green')
    b_col = cols.get('b') or cols.get('blue')
    if not (id_col and r_col and g_col and b_col):
        return {}

    cmap: dict[int, tuple[int, int, int]] = {}
    for _, row in df.iterrows():
        try:
            rid = int(row[id_col])
            cmap[rid] = (int(row[r_col]), int(row[g_col]), int(row[b_col]))
        except Exception:
            continue
    return cmap
