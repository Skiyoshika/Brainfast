"""Unit tests for Phase γ class-prior closed-loop storage.

Phase γ accumulates per-sample-class landmark corrections so the 4th+
sample of the same class starts with a warm-start set of landmarks rather
than requiring the user to re-do every correction from scratch.

The store is a running-mean merge indexed by (z, atlas_y, atlas_x) within a
merge radius. Samples below the min-samples threshold do not yet expose a
usable prior — we flag that with ``load_prior`` returning ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _pairs_from(tuples):
    """Helper: turn (z, ay, ax, ry, rx) tuples into LandmarkPair instances."""
    from project.scripts.liquify_3d import LandmarkPair

    return [
        LandmarkPair(z=z, atlas=(ay, ax), real=(ry, rx)) for (z, ay, ax, ry, rx) in tuples
    ]


# ---------------------------------------------------------------------------
# ClassPriorStore: create / update / load
# ---------------------------------------------------------------------------


def test_update_accepts_iterator_and_preserves_pair_count(tmp_path):
    """Task 3 — ``ClassPriorStore.update`` was logging ``len(list(pairs))``
    *after* already iterating once, so iterator inputs recorded pair_count=0
    in sample_log.jsonl even though the pairs were consumed normally. The
    fix converts ``pairs`` to a list once at the top.
    """
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    pair_tuples = [(10, 100.0, 100.0, 103.0, 99.0), (10, 200.0, 200.0, 205.0, 198.0)]
    # Pass as a GENERATOR, not a list — regression trigger.
    store.update(
        sample_id="iter_sample",
        pairs=(p for p in _pairs_from(pair_tuples)),
    )
    log_path = tmp_path / "ChATe27" / "sample_log.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["pair_count"] == 2


def test_store_creates_class_subdir_on_update(tmp_path):
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    pairs = _pairs_from([(100, 50, 80, 55, 82)])
    store.update(sample_id="sample35", pairs=pairs)

    assert (tmp_path / "ChATe27" / "landmark_prior.csv").exists()
    assert (tmp_path / "ChATe27" / "sample_log.jsonl").exists()


def test_first_sample_yields_single_mean_entry(tmp_path):
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    pairs = _pairs_from([(100, 50, 80, 55, 82)])
    store.update(sample_id="sample35", pairs=pairs)

    entries = store.entries()
    assert len(entries) == 1
    e = entries[0]
    assert e.z == 100
    assert e.atlas == (50.0, 80.0)
    assert e.mean_dy == pytest.approx(5.0)   # 55-50
    assert e.mean_dx == pytest.approx(2.0)   # 82-80
    assert e.n == 1
    assert store.sample_count() == 1


def test_second_sample_same_voxel_is_running_mean_merged(tmp_path):
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    store.update("s1", _pairs_from([(100, 50, 80, 55, 82)]))  # dy=5, dx=2
    store.update("s2", _pairs_from([(100, 50, 80, 57, 86)]))  # dy=7, dx=6

    entries = store.entries()
    assert len(entries) == 1, "two pairs at same atlas voxel must merge"
    e = entries[0]
    assert e.n == 2
    assert e.mean_dy == pytest.approx(6.0)   # (5+7)/2
    assert e.mean_dx == pytest.approx(4.0)   # (2+6)/2
    assert store.sample_count() == 2


def test_far_apart_pairs_become_separate_entries(tmp_path):
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    # Two pairs well outside merge_radius=8 default
    store.update("s1", _pairs_from([(100, 50, 80, 55, 82), (100, 200, 200, 204, 201)]))

    entries = store.entries()
    assert len(entries) == 2
    zs = sorted((e.atlas[0], e.atlas[1]) for e in entries)
    assert zs == [(50.0, 80.0), (200.0, 200.0)]


def test_different_z_never_merges(tmp_path):
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    store.update("s1", _pairs_from([(100, 50, 80, 55, 82)]))
    store.update("s2", _pairs_from([(200, 50, 80, 55, 82)]))

    entries = store.entries()
    assert len(entries) == 2


def test_sample_log_appends_jsonl(tmp_path):
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    store.update("s1", _pairs_from([(100, 50, 80, 55, 82)]), metrics={"NCC": 0.3})
    store.update("s2", _pairs_from([(100, 50, 80, 57, 86)]), metrics={"NCC": 0.4})

    log_path = tmp_path / "ChATe27" / "sample_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["sample_id"] == "s1"
    assert first["pair_count"] == 1
    assert first["metrics"] == {"NCC": 0.3}


def test_repeated_update_for_same_sample_is_idempotent(tmp_path):
    """A sample can be saved more than once from the UI, but it should only
    contribute once to the learned prior and readiness threshold.
    """
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    store.update("same_sample", _pairs_from([(100, 50, 80, 55, 82)]))
    store.update("same_sample", _pairs_from([(100, 50, 80, 65, 92)]))

    entries = store.entries()
    assert len(entries) == 1
    assert entries[0].n == 1
    assert entries[0].mean_dy == pytest.approx(5.0)
    assert entries[0].mean_dx == pytest.approx(2.0)
    assert store.sample_count() == 1


def test_sample_count_ignores_qc_done_history_rows(tmp_path):
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    store.update("sample35", _pairs_from([(100, 50, 80, 55, 82)]))
    log_path = tmp_path / "ChATe27" / "sample_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "qc_done",
                    "sample_id": "sample35",
                    "pair_count": 1,
                    "metrics": {"NCC": 0.55},
                }
            )
            + "\n"
        )

    assert store.sample_count() == 1


def test_persisted_entries_survive_reopen(tmp_path):
    from project.scripts.class_prior import ClassPriorStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    store.update("s1", _pairs_from([(100, 50, 80, 55, 82)]))
    store.update("s2", _pairs_from([(100, 50, 80, 57, 86)]))

    reopened = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    entries = reopened.entries()
    assert len(entries) == 1
    assert entries[0].n == 2
    assert reopened.sample_count() == 2


# ---------------------------------------------------------------------------
# load_prior: confidence gate + apply as warm-start
# ---------------------------------------------------------------------------


def test_load_prior_below_min_samples_returns_none(tmp_path):
    from project.scripts.class_prior import ClassPriorStore, load_prior

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    store.update("s1", _pairs_from([(100, 50, 80, 55, 82)]))

    # Default min_samples=3; should refuse to apply with only 1
    assert load_prior("ChATe27", priors_root=tmp_path) is None


def test_load_prior_meets_threshold_returns_store(tmp_path):
    from project.scripts.class_prior import ClassPriorStore, load_prior

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    for i in range(3):
        store.update(f"s{i}", _pairs_from([(100, 50, 80, 55 + i, 82 + i)]))

    prior = load_prior("ChATe27", priors_root=tmp_path)
    assert prior is not None
    assert prior.sample_count() == 3


def test_apply_as_warm_start_populates_landmarks_csv(tmp_path):
    from project.scripts.class_prior import ClassPriorStore
    from project.scripts.liquify_3d import LandmarkStore

    # Build a prior with 3 samples of one pair
    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    for i in range(3):
        store.update(f"s{i}", _pairs_from([(100, 50, 80, 55 + i, 82 + i)]))

    target_csv = tmp_path / "target_job" / "landmarks_3d.csv"
    store.apply_as_warm_start(target_csv)

    landmarks = LandmarkStore(target_csv).list_pairs()
    assert len(landmarks) == 1
    lm = landmarks[0]
    assert lm.z == 100
    assert lm.atlas == (50.0, 80.0)
    # Mean real = atlas + mean_displacement = (50+1, 80+1) = (51, 81); but we
    # averaged 55,56,57 and 82,83,84 so mean_dy=1? no: 55-50=5, 56-50=6, 57-50=7 → mean_dy=6
    assert lm.real == pytest.approx((56.0, 83.0))


def test_apply_as_warm_start_skips_when_target_already_has_pairs(tmp_path):
    """Do not clobber landmarks the user has already laid down manually."""
    from project.scripts.class_prior import ClassPriorStore
    from project.scripts.liquify_3d import LandmarkStore

    store = ClassPriorStore(class_name="ChATe27", priors_root=tmp_path)
    for i in range(3):
        store.update(f"s{i}", _pairs_from([(100, 50, 80, 55 + i, 82 + i)]))

    target_csv = tmp_path / "target_job" / "landmarks_3d.csv"
    target = LandmarkStore(target_csv)
    target.add_pair(z=200, real=(100, 100), atlas=(99, 99))

    # Attempting to apply warm-start to a non-empty store should refuse without
    # force=True so we never overwrite manual corrections.
    with pytest.raises(ValueError, match="existing"):
        store.apply_as_warm_start(target_csv)

    # force=True replaces the contents with prior-derived pairs
    store.apply_as_warm_start(target_csv, force=True)
    landmarks = LandmarkStore(target_csv).list_pairs()
    assert len(landmarks) == 1
    assert landmarks[0].z == 100
