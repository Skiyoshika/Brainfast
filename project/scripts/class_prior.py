"""Phase γ: class-prior closed-loop storage.

Accumulates user-provided landmark corrections (from Phase β) across samples
of the same experimental class (e.g. ``ChATe27``, ``PVe3``) so that future
samples of the same class start with a pre-populated set of landmarks that
already matches the biases seen in prior runs. After ~3+ samples the new
sample's automatic registration should land close enough to the target that
little-to-no manual correction is needed.

Storage layout::

    <priors_root>/
      <class_name>/
        landmark_prior.csv   # running-mean entries
        sample_log.jsonl     # one line per contributing sample

Running-mean merge rule
-----------------------
An entry is keyed by ``(z, atlas_y, atlas_x)``. A new pair is *merged* into
the nearest existing entry if its atlas coord is within ``merge_radius_vx``
of the entry's atlas coord **and** the z matches exactly; otherwise a new
entry is created. This keeps "the same anatomical point" aggregated across
samples even if the user clicked a voxel or two off.

Confidence gate
---------------
``load_prior(class_name)`` returns ``None`` while ``sample_count`` is below
``MIN_SAMPLES_FOR_APPLY`` (default 3) so we do not warm-start a new sample
from a single biased correction. This matches the plan's "require ≥ 3
samples before auto-applying" risk mitigation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

MIN_SAMPLES_FOR_APPLY: int = 3
MERGE_RADIUS_VX: float = 8.0


@dataclass(frozen=True)
class PriorEntry:
    """One aggregated landmark location in the class prior.

    ``mean_dy``/``mean_dx`` are the average displacement vectors pointing
    from the *atlas* position to the *real* position observed across ``n``
    contributing samples. Applying this prior as a warm-start for a new
    sample means seeding that sample's landmark store with::

        atlas = (atlas_y, atlas_x)
        real  = (atlas_y + mean_dy, atlas_x + mean_dx)
    """

    z: int
    atlas: tuple[float, float]
    mean_dy: float
    mean_dx: float
    n: int


class ClassPriorStore:
    """Per-class aggregator of landmark corrections across samples."""

    _ENTRY_COLUMNS = (
        "z",
        "atlas_y",
        "atlas_x",
        "sum_dy",
        "sum_dx",
        "sum_sq_dy",
        "sum_sq_dx",
        "n",
    )

    def __init__(
        self,
        class_name: str,
        priors_root: Path | str,
        *,
        merge_radius_vx: float = MERGE_RADIUS_VX,
    ) -> None:
        self.class_name = str(class_name)
        self.priors_root = Path(priors_root)
        self._class_dir = self.priors_root / self.class_name
        self._entry_csv = self._class_dir / "landmark_prior.csv"
        self._sample_log = self._class_dir / "sample_log.jsonl"
        self._merge_radius = float(merge_radius_vx)

    # ----- Inspection -----

    @property
    def class_dir(self) -> Path:
        return self._class_dir

    def entries(self) -> list[PriorEntry]:
        if not self._entry_csv.exists():
            return []
        out: list[PriorEntry] = []
        with self._entry_csv.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                n = int(row["n"])
                sum_dy = float(row["sum_dy"])
                sum_dx = float(row["sum_dx"])
                out.append(
                    PriorEntry(
                        z=int(row["z"]),
                        atlas=(float(row["atlas_y"]), float(row["atlas_x"])),
                        mean_dy=sum_dy / max(n, 1),
                        mean_dx=sum_dx / max(n, 1),
                        n=n,
                    )
                )
        return out

    def sample_count(self) -> int:
        """Number of distinct samples that have contributed to this prior."""
        return len(self._contributing_sample_ids())

    # ----- Update -----

    def update(
        self,
        sample_id: str,
        pairs,
        *,
        metrics: dict | None = None,
    ) -> None:
        """Merge one sample's landmark pairs into the running-mean store.

        Parameters
        ----------
        sample_id
            Unique identifier for the contributing sample (stored in the
            sample log).
        pairs
            Iterable of ``liquify_3d.LandmarkPair`` objects.
        metrics
            Optional metrics dict recorded alongside the sample log entry
            (e.g. post-registration NCC/Dice from the run).
        """
        self._class_dir.mkdir(parents=True, exist_ok=True)
        # Iterator-safe: materialize once. The sample_log count at the end
        # previously read ``len(list(pairs))`` AFTER the merge loop had
        # already exhausted a generator input, logging 0 instead of N.
        pair_list = list(pairs)
        if self._has_prior_sample(sample_id):
            self._append_sample_log(
                sample_id=sample_id,
                pair_count=len(pair_list),
                metrics=metrics,
                kind="prior_update_duplicate_ignored",
            )
            return
        raw_entries = self._load_raw_entries()
        for pair in pair_list:
            dy = pair.real[0] - pair.atlas[0]
            dx = pair.real[1] - pair.atlas[1]
            idx = self._find_merge_target(
                raw_entries, z=pair.z, atlas_y=pair.atlas[0], atlas_x=pair.atlas[1]
            )
            if idx is None:
                raw_entries.append(
                    {
                        "z": int(pair.z),
                        "atlas_y": float(pair.atlas[0]),
                        "atlas_x": float(pair.atlas[1]),
                        "sum_dy": dy,
                        "sum_dx": dx,
                        "sum_sq_dy": dy * dy,
                        "sum_sq_dx": dx * dx,
                        "n": 1,
                    }
                )
            else:
                e = raw_entries[idx]
                e["sum_dy"] += dy
                e["sum_dx"] += dx
                e["sum_sq_dy"] += dy * dy
                e["sum_sq_dx"] += dx * dx
                e["n"] += 1
        self._write_raw_entries(raw_entries)
        self._append_sample_log(
            sample_id=sample_id,
            pair_count=len(pair_list),
            metrics=metrics,
            kind="prior_update",
        )

    # ----- Warm-start application -----

    def apply_as_warm_start(
        self,
        target_csv: Path | str,
        *,
        force: bool = False,
    ) -> int:
        """Write the prior's mean landmarks into *target_csv* as an initial
        set of LandmarkPair rows. Returns the number of pairs written.

        Refuses to run when *target_csv* already has pairs unless
        ``force=True`` — prior should never silently clobber a user's manual
        corrections.
        """
        from project.scripts.liquify_3d import LandmarkStore

        target_csv = Path(target_csv)
        lm = LandmarkStore(target_csv)
        if lm.list_pairs() and not force:
            raise ValueError(
                f"target landmarks CSV {target_csv} already has existing pairs; "
                "pass force=True to overwrite"
            )
        if force:
            lm.clear()

        written = 0
        for e in self.entries():
            real = (e.atlas[0] + e.mean_dy, e.atlas[1] + e.mean_dx)
            lm.add_pair(z=e.z, atlas=e.atlas, real=real)
            written += 1
        return written

    # ----- Internals -----

    def _load_raw_entries(self) -> list[dict]:
        if not self._entry_csv.exists():
            return []
        with self._entry_csv.open("r", newline="", encoding="utf-8") as fh:
            return [
                {
                    "z": int(row["z"]),
                    "atlas_y": float(row["atlas_y"]),
                    "atlas_x": float(row["atlas_x"]),
                    "sum_dy": float(row["sum_dy"]),
                    "sum_dx": float(row["sum_dx"]),
                    "sum_sq_dy": float(row["sum_sq_dy"]),
                    "sum_sq_dx": float(row["sum_sq_dx"]),
                    "n": int(row["n"]),
                }
                for row in csv.DictReader(fh)
            ]

    def _write_raw_entries(self, entries: list[dict]) -> None:
        # Unique-tmpfile + retry-on-PermissionError so concurrent writers
        # (e.g. a user double-clicking "Save job → class prior") don't
        # collide on a shared .tmp filename the way Windows hates.
        import os as _os
        import tempfile as _tempfile
        import time as _time

        self._entry_csv.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = _tempfile.mkstemp(
            prefix=self._entry_csv.name + ".",
            suffix=".tmp",
            dir=str(self._entry_csv.parent),
        )
        tmp = Path(tmp_name)
        try:
            with _os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(self._ENTRY_COLUMNS)
                for e in entries:
                    w.writerow(
                        [
                            e["z"],
                            e["atlas_y"],
                            e["atlas_x"],
                            e["sum_dy"],
                            e["sum_dx"],
                            e["sum_sq_dy"],
                            e["sum_sq_dx"],
                            e["n"],
                        ]
                    )
            last_error: OSError | None = None
            for attempt in range(8):
                try:
                    tmp.replace(self._entry_csv)
                    last_error = None
                    break
                except PermissionError as exc:
                    last_error = exc
                    _time.sleep(0.01 * (attempt + 1))
            if last_error is not None:
                raise last_error
        finally:
            if tmp.exists():
                tmp.unlink()

    def _find_merge_target(
        self,
        entries: list[dict],
        *,
        z: int,
        atlas_y: float,
        atlas_x: float,
    ) -> int | None:
        best_i: int | None = None
        best_d2 = self._merge_radius * self._merge_radius
        for i, e in enumerate(entries):
            if int(e["z"]) != int(z):
                continue
            dy = float(e["atlas_y"]) - float(atlas_y)
            dx = float(e["atlas_x"]) - float(atlas_x)
            d2 = dy * dy + dx * dx
            if d2 <= best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _append_sample_log(
        self,
        *,
        sample_id: str,
        pair_count: int,
        metrics: dict | None,
        kind: str,
    ) -> None:
        rec = {
            "kind": str(kind),
            "sample_id": str(sample_id),
            "pair_count": int(pair_count),
            "metrics": dict(metrics) if metrics else None,
        }
        with self._sample_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _iter_sample_log_records(self):
        if not self._sample_log.exists():
            return
        with self._sample_log.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    yield rec

    @staticmethod
    def _sample_log_kind(rec: dict) -> str:
        kind = str(rec.get("kind") or "").strip()
        if kind:
            return kind
        if "qc_done_ts" in rec or "qc_note" in rec:
            return "qc_done"
        return "prior_update"

    def _contributing_sample_ids(self) -> set[str]:
        sample_ids: set[str] = set()
        for rec in self._iter_sample_log_records() or ():
            if self._sample_log_kind(rec) != "prior_update":
                continue
            try:
                pair_count = int(rec.get("pair_count") or 0)
            except (TypeError, ValueError):
                pair_count = 0
            if pair_count <= 0:
                continue
            sample_id = str(rec.get("sample_id") or "").strip()
            if sample_id:
                sample_ids.add(sample_id)
        return sample_ids

    def _has_prior_sample(self, sample_id: str) -> bool:
        sample_id = str(sample_id).strip()
        return bool(sample_id) and sample_id in self._contributing_sample_ids()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_prior(
    class_name: str,
    priors_root: Path | str,
    *,
    min_samples: int = MIN_SAMPLES_FOR_APPLY,
) -> ClassPriorStore | None:
    """Return the class's prior store only when it has enough samples to apply.

    Returning ``None`` signals the caller should not warm-start a new sample
    with this class's prior yet — either the class has never been corrected
    or has fewer than ``min_samples`` contributing runs (so the running mean
    is still too noisy to help).
    """
    store = ClassPriorStore(class_name=class_name, priors_root=priors_root)
    return store if store.sample_count() >= int(min_samples) else None
