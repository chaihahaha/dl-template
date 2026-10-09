"""Integration tests for the optional PyTorch Lightning logger."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from lightning.pytorch import LightningModule, Trainer  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from dlt.integrations.lightning import DLTLogger  # noqa: E402
from dlt.yaml_logging import load_log  # noqa: E402


class TinyModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = ((self(x) - y) ** 2).mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_manual_logger_api(tmp_path: Path) -> None:
    logger = DLTLogger("manual", save_dir=tmp_path / "runs", save_every=0)
    assert logger.name == "manual"
    assert logger.version and logger.save_dir and logger.log_dir
    assert logger.experiment is logger._logger

    logger.log_hyperparams({"lr": 0.1, "nested": {"a": 1}})
    logger.log_metrics({"loss": 0.5}, step=3)
    logger.finalize("success")

    data = load_log(logger._logger.path)
    assert data["config"] == {"lr": 0.1, "nested": {"a": 1}}
    assert data["metrics"]["loss"] == [0.5]
    assert data["latest"]["step"] == 3
    assert any("finalize" in line for line in data["info"])


def test_trainer_writes_dlt_log(tmp_path: Path) -> None:
    logger = DLTLogger("tiny", save_dir=tmp_path / "runs", save_every=0, async_save=True)
    trainer = Trainer(
        logger=logger,
        max_epochs=1,
        limit_train_batches=2,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        num_sanity_val_steps=0,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(16, 4), torch.randn(16, 1)), batch_size=8
    )
    trainer.fit(TinyModel(), train_dataloaders=loader)

    data = load_log(logger._logger.path)
    assert "train_loss" in data["metrics"]
    assert len(data["metrics"]["train_loss"]) == 2
    # finalize() closes the logger; the async write must have been flushed.
    assert any("finalize" in line for line in data["info"])


def test_run_directory_layout(tmp_path: Path) -> None:
    logger = DLTLogger("layout", save_dir=tmp_path / "runs", config={"lr": 1e-3})
    run_dir = Path(logger.save_dir).parent
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "logs").is_dir()
    logger.finalize("success")
