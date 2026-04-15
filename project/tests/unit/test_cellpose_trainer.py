"""Tests for cellpose_trainer.py — training wrapper module."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestTrainerState:
    def test_initial_state_is_idle(self):
        from project.scripts.cellpose_trainer import TrainingState
        state = TrainingState()
        assert state.status == "idle"
        assert state.epoch == 0
        assert state.total_epochs == 0

    def test_state_to_dict(self):
        from project.scripts.cellpose_trainer import TrainingState
        state = TrainingState()
        state.status = "training"
        state.epoch = 10
        state.total_epochs = 100
        state.train_loss = 0.5
        state.test_loss = 0.6
        d = state.to_dict()
        assert d["status"] == "training"
        assert d["epoch"] == 10
        assert d["totalEpochs"] == 100
        assert d["trainLoss"] == 0.5
        assert d["testLoss"] == 0.6

    def test_state_reset(self):
        from project.scripts.cellpose_trainer import TrainingState
        state = TrainingState()
        state.status = "training"
        state.epoch = 50
        state.reset()
        assert state.status == "idle"
        assert state.epoch == 0


class TestLoadTrainingData:
    def test_loads_images_and_masks(self, tmp_path):
        from tifffile import imwrite

        from project.scripts.cellpose_trainer import load_training_data

        # Create 3 image/mask pairs
        for i in range(3):
            img = np.random.randint(0, 65535, (32, 32), dtype=np.uint16)
            mask = np.random.randint(0, 5, (32, 32), dtype=np.uint16)
            imwrite(str(tmp_path / f"img_{i}.tif"), img)
            imwrite(str(tmp_path / f"img_{i}_masks.tif"), mask)

        images, labels = load_training_data(tmp_path)
        assert len(images) == 3
        assert len(labels) == 3
        assert images[0].shape == (32, 32)

    def test_empty_dir_raises(self, tmp_path):
        from project.scripts.cellpose_trainer import TrainingError, load_training_data
        with pytest.raises(TrainingError, match="No training"):
            load_training_data(tmp_path)

    def test_insufficient_data_raises(self, tmp_path):
        from tifffile import imwrite

        from project.scripts.cellpose_trainer import TrainingError, load_training_data

        # Only 1 pair — need at least 2
        imwrite(str(tmp_path / "a.tif"), np.zeros((8, 8), dtype=np.uint16))
        imwrite(str(tmp_path / "a_masks.tif"), np.ones((8, 8), dtype=np.uint16))

        with pytest.raises(TrainingError, match="at least 2"):
            load_training_data(tmp_path)


class TestStartTraining:
    def test_start_sets_status_to_training(self, tmp_path):
        from project.scripts.cellpose_trainer import CellposeTrainer

        trainer = CellposeTrainer()

        # Create enough training data
        for i in range(3):
            img = np.random.randint(0, 255, (16, 16), dtype=np.uint16)
            mask = np.random.randint(0, 3, (16, 16), dtype=np.uint16)
            from tifffile import imwrite
            imwrite(str(tmp_path / f"s{i}.tif"), img)
            imwrite(str(tmp_path / f"s{i}_masks.tif"), mask)

        # Mock the actual training
        with patch("project.scripts.cellpose_trainer._run_training") as mock_train:
            mock_train.return_value = (str(tmp_path / "model.pth"), [0.5, 0.3], [0.6, 0.4])
            trainer.start(
                training_dir=tmp_path,
                base_model="cyto3",
                model_name="test_model",
                n_epochs=10,
            )
            # Give thread a moment to start
            time.sleep(0.3)
            status = trainer.state.to_dict()
            # Should be training or completed (fast mock)
            assert status["status"] in ("training", "completed")

    def test_cannot_start_while_training(self, tmp_path):
        from project.scripts.cellpose_trainer import CellposeTrainer, TrainingError

        trainer = CellposeTrainer()
        trainer.state.status = "training"

        with pytest.raises(TrainingError, match="already"):
            trainer.start(tmp_path, "cyto3", "model", 10)

    def test_cancel_sets_cancelled(self):
        from project.scripts.cellpose_trainer import CellposeTrainer

        trainer = CellposeTrainer()
        trainer.state.status = "training"
        trainer.cancel()
        assert trainer.state.status == "cancelled"


class TestUpdateConfigModel:
    def test_updates_config_file(self, tmp_path):
        import json

        from project.scripts.cellpose_trainer import _update_config_primary_model

        # Create a mock config file
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "detection": {"primary_model": "cyto3"},
            "other": "value",
        }))

        # Patch server_context to point to our config
        import project.frontend.server_context as ctx
        original = ctx.run_state.get("config_path")
        ctx.run_state["config_path"] = str(config_path)
        try:
            _update_config_primary_model("brainfast_v1")

            updated = json.loads(config_path.read_text())
            assert updated["detection"]["primary_model"] == "brainfast_v1"
            assert updated["other"] == "value"  # other keys preserved
        finally:
            if original is not None:
                ctx.run_state["config_path"] = original
            else:
                ctx.run_state.pop("config_path", None)


class TestClearModelCache:
    def test_clears_cache(self):
        from project.scripts.cellpose_trainer import _clear_model_cache
        from project.scripts.detect import _CELLPOSE_MODEL_CACHE

        # Add a dummy entry
        _CELLPOSE_MODEL_CACHE[("test", False)] = "dummy"
        assert len(_CELLPOSE_MODEL_CACHE) > 0

        _clear_model_cache()
        assert len(_CELLPOSE_MODEL_CACHE) == 0
