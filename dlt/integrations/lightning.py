"""PyTorch Lightning integration (optional).

Drop-in ``LightningLogger`` so a Lightning ``Trainer`` writes the same
crash-safe YAML log as the rest of this template::

    from lightning.pytorch import Trainer
    from dlt.integrations.lightning import DLTLogger

    trainer = Trainer(
        logger=DLTLogger("resnet50", save_dir="runs", save_every=100, async_save=True),
        # These are the Trainer features that actually accelerate training:
        accelerator="gpu", devices=4, strategy="ddp",
        precision="16-mixed",          # or "bf16-mixed"
        accumulate_grad_batches=2,
        # torch>=2.0, opt-in:
        # compile=True,
    )

Notes
-----
* ``dlt`` itself never touches the training step; Lightning's ``Trainer`` is
  what provides DDP/FSDP, mixed precision, gradient accumulation, ``torch.compile``
  and gradient clipping.  This logger only gives the ``Trainer`` a place to
  send metrics/hyper-parameters.
* ``log_metrics`` is called by Lightning on the *main* process only, so it does
  not introduce per-rank I/O; with ``async_save=True`` the YAML serialisation
  happens off the training thread.
* ``finalize`` closes the logger, which flushes any pending async write.

Requires ``lightning`` (or legacy ``pytorch_lightning``); both are tried.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from ..naming import ExperimentNamer
from ..yaml_logging import YamlLogger

try:  # Lightning >= 2.0
    from lightning.pytorch.loggers import Logger as _LightningLogger
except Exception:  # pragma: no cover - legacy import path
    try:
        from pytorch_lightning.loggers import Logger as _LightningLogger  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "DLTLogger requires `lightning` or `pytorch_lightning` to be installed"
        ) from exc

__all__ = ["DLTLogger"]


def _as_mapping(params: Any) -> dict[str, Any]:
    """Normalise the many shapes Lightning uses for hyper-parameters."""
    if params is None:
        return {}
    if isinstance(params, Mapping):
        return dict(params)
    if hasattr(params, "items"):
        try:
            return dict(params.items())
        except Exception:
            pass
    if hasattr(params, "__dict__"):
        return dict(vars(params))
    if isinstance(params, (list, tuple)):
        merged: dict[str, Any] = {}
        for item in params:
            merged.update(_as_mapping(item))
        return merged
    return {"params": params}


class DLTLogger(_LightningLogger):
    """Lightning logger backed by :class:`dlt.yaml_logging.YamlLogger`."""

    def __init__(
        self,
        name: str = "lightning",
        *,
        save_dir: str | os.PathLike[str] = "runs",
        config: Mapping[str, Any] | None = None,
        run: Any | None = None,
        **logger_kwargs: Any,
    ) -> None:
        if run is None:
            run = ExperimentNamer(save_dir, name=name).new_run(config)
        self._logger = YamlLogger(name, run=run, config=config, **logger_kwargs)

    # -- Lightning Logger API -------------------------------------------- #
    @property
    def name(self) -> str:
        return self._logger.name

    @property
    def version(self) -> str:
        return self._logger.experiment_id

    @property
    def save_dir(self) -> str:
        return os.fspath(self._logger.path.parent)

    @property
    def log_dir(self) -> str:
        return os.fspath(self._logger.path.parent)

    @property
    def experiment(self) -> YamlLogger:
        """Some Lightning utilities expect ``logger.experiment`` to exist."""
        return self._logger

    # -- what Lightning calls -------------------------------------------- #
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.set_config(_as_mapping(params))

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        payload: dict[str, Any] = dict(metrics)
        if step is not None:
            payload.setdefault("step", int(step))
        self._logger.log(payload)

    def save(self) -> None:
        self._logger.save()

    def finalize(self, status: str) -> None:
        self._logger.add_info(f"lightning finalize: {status}")
        self._logger.close()

    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:
        path = getattr(checkpoint_callback, "best_model_path", "") or getattr(
            checkpoint_callback, "last_model_path", ""
        )
        if path:
            self._logger.add_artifact(path)
