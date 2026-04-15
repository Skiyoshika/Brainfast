"""cellpose_trainer.py — Background Cellpose model training wrapper.

Manages the full training lifecycle:
  1. Load training data (image/mask pairs from a directory)
  2. Run train_seg() in a background thread with progress callbacks
  3. Register the trained model in Cellpose's model registry
  4. Clear the detection model cache so new model is used immediately
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import imread

_log = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when training cannot start or fails."""


@dataclass
class TrainingState:
    """Shared mutable state for the training thread."""

    status: str = "idle"  # idle | training | completed | failed | cancelled
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    test_loss: float = 0.0
    loss_history: list[list[float]] = field(default_factory=list)
    model_path: str = ""
    model_name: str = ""
    error: str = ""
    started_at: float = 0.0

    def reset(self) -> None:
        self.status = "idle"
        self.epoch = 0
        self.total_epochs = 0
        self.train_loss = 0.0
        self.test_loss = 0.0
        self.loss_history = []
        self.model_path = ""
        self.model_name = ""
        self.error = ""
        self.started_at = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": self.status,
            "epoch": self.epoch,
            "totalEpochs": self.total_epochs,
            "trainLoss": round(self.train_loss, 4),
            "testLoss": round(self.test_loss, 4),
            "lossHistory": self.loss_history,
            "modelPath": self.model_path,
            "modelName": self.model_name,
            "error": self.error,
        }
        if self.started_at > 0 and self.epoch > 0 and self.total_epochs > 0:
            elapsed = time.time() - self.started_at
            per_epoch = elapsed / self.epoch
            remaining = per_epoch * (self.total_epochs - self.epoch)
            mins, secs = divmod(int(remaining), 60)
            d["estimatedTimeRemaining"] = f"{mins}m {secs}s"
        return d


def load_training_data(training_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load image/mask pairs from a training directory.

    Expects Cellpose convention: ``{name}.tif`` + ``{name}_masks.tif``.
    Returns (images, labels) lists.
    """
    mask_files = sorted(training_dir.glob("*_masks.tif"))
    if not mask_files:
        raise TrainingError(f"No training data found in {training_dir}")

    images = []
    labels = []
    for mf in mask_files:
        stem = mf.name.replace("_masks.tif", "")
        img_path = training_dir / f"{stem}.tif"
        if not img_path.exists():
            _log.warning("Mask file %s has no matching image, skipping", mf.name)
            continue
        images.append(imread(str(img_path)))
        labels.append(imread(str(mf)).astype(np.int32))

    if len(images) < 2:
        raise TrainingError(f"Need at least 2 image/mask pairs for training, found {len(images)}")

    _log.info("Loaded %d training pairs from %s", len(images), training_dir)
    return images, labels


def _run_training(
    base_model: str,
    images: list[np.ndarray],
    labels: list[np.ndarray],
    n_epochs: int,
    model_name: str,
    save_path: str,
    state: TrainingState,
    use_gpu: bool = True,
) -> tuple[str, list[float], list[float]]:
    """Run Cellpose training. Called in background thread."""
    from cellpose.models import CellposeModel
    from cellpose.train import train_seg

    # Load base model
    model = CellposeModel(model_type=base_model, gpu=use_gpu)

    # Split 80/20 train/test
    n = len(images)
    split = max(1, int(n * 0.8))
    train_data, test_data = images[:split], images[split:] or [images[-1]]
    train_labels, test_labels = labels[:split], labels[split:] or [labels[-1]]

    state.status = "training"
    state.total_epochs = n_epochs
    state.started_at = time.time()

    # Run training
    model_path, train_losses, test_losses = train_seg(
        net=model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        n_epochs=n_epochs,
        learning_rate=1e-5,
        weight_decay=0.1,
        batch_size=1,
        save_path=save_path,
        model_name=model_name,
        min_train_masks=1,
    )

    return str(model_path), list(train_losses), list(test_losses)


def _clear_model_cache() -> None:
    """Clear the Cellpose model cache in detect.py so the new model is loaded."""
    try:
        from project.scripts.detect import _CELLPOSE_MODEL_CACHE

        _CELLPOSE_MODEL_CACHE.clear()
        _log.info("Cleared Cellpose model cache")
    except ImportError:
        try:
            from scripts.detect import _CELLPOSE_MODEL_CACHE

            _CELLPOSE_MODEL_CACHE.clear()
        except ImportError:
            _log.warning("Could not clear model cache — detect module not found")


def _register_model(model_path: str) -> None:
    """Register the trained model in Cellpose's user model registry."""
    try:
        from cellpose.io import add_model

        add_model(model_path)
        _log.info("Registered model: %s", model_path)
    except Exception as exc:
        _log.warning("Could not register model: %s", exc)


def _update_config_primary_model(model_name: str) -> None:
    """Update the active config's primary_model to the newly trained model."""
    import json

    try:
        import project.frontend.server_context as ctx

        config_path_str = ctx.run_state.get("config_path")
    except ImportError:
        config_path_str = None

    if not config_path_str:
        # Fallback to template
        candidates = []
        try:
            import project.frontend.server_context as ctx

            candidates.append(ctx.PROJECT_ROOT / "configs" / "run_config.template.json")
        except Exception:
            pass
        for p in candidates:
            if p.exists():
                config_path_str = str(p)
                break

    if not config_path_str:
        _log.warning("No config file found to update primary_model")
        return

    config_path = Path(config_path_str)
    if not config_path.exists():
        _log.warning("Config file not found: %s", config_path)
        return

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if "detection" not in cfg:
            cfg["detection"] = {}
        old_model = cfg["detection"].get("primary_model", "")
        cfg["detection"]["primary_model"] = model_name
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        _log.info("Updated config %s: primary_model %s -> %s", config_path, old_model, model_name)
    except Exception as exc:
        _log.warning("Failed to update config primary_model: %s", exc)


class CellposeTrainer:
    """Manages background Cellpose training with progress tracking."""

    def __init__(self) -> None:
        self.state = TrainingState()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(
        self,
        training_dir: Path,
        base_model: str,
        model_name: str,
        n_epochs: int = 100,
        use_gpu: bool = True,
    ) -> None:
        """Start training in a background thread."""
        with self._lock:
            if self.state.status == "training":
                raise TrainingError("Training is already in progress")

        # Validate data before starting thread
        images, labels = load_training_data(training_dir)

        save_path = str(training_dir / "models")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        self.state.reset()
        self.state.status = "training"
        self.state.model_name = model_name
        self.state.total_epochs = n_epochs

        def _train_thread():
            try:
                model_path, train_losses, test_losses = _run_training(
                    base_model=base_model,
                    images=images,
                    labels=labels,
                    n_epochs=n_epochs,
                    model_name=model_name,
                    save_path=save_path,
                    state=self.state,
                    use_gpu=use_gpu,
                )

                if self.state.status == "cancelled":
                    return

                # Update final state
                self.state.model_path = model_path
                if train_losses:
                    self.state.train_loss = train_losses[-1]
                    self.state.loss_history = [
                        [i, tl, tel]
                        for i, (tl, tel) in enumerate(
                            zip(train_losses, test_losses or train_losses, strict=False)
                        )
                    ]
                if test_losses:
                    self.state.test_loss = test_losses[-1]

                # Register model, clear cache, and update active config
                _register_model(model_path)
                _clear_model_cache()
                _update_config_primary_model(model_name)

                self.state.status = "completed"
                _log.info(
                    "Training completed: %s (train_loss=%.4f, test_loss=%.4f)",
                    model_name,
                    self.state.train_loss,
                    self.state.test_loss,
                )

            except Exception as exc:
                self.state.status = "failed"
                self.state.error = str(exc)
                _log.error("Training failed: %s", exc)

        self._thread = threading.Thread(target=_train_thread, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation of the current training."""
        if self.state.status == "training":
            self.state.status = "cancelled"
            _log.info("Training cancellation requested")

    def get_status(self) -> dict[str, Any]:
        """Return current training state as a dict."""
        return self.state.to_dict()


# Module-level singleton for the Flask app to use
_trainer = CellposeTrainer()


def get_trainer() -> CellposeTrainer:
    """Return the module-level trainer singleton."""
    return _trainer
