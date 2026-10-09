"""PyTorch Lightning + dlt example (optional dependency).

Shows the Trainer features that actually accelerate training, together with the
``DLTLogger`` so the run gets a crash-safe YAML log and a named run directory::

    pip install ".[lightning]"
    python examples/lightning_demo.py

The default configuration runs on CPU so it works anywhere.  On a real machine
the commented options below switch on the usual accelerations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from dlt.integrations.lightning import DLTLogger


class MLP(LightningModule):
    def __init__(self, d_in: int = 16, d_hidden: int = 64) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)


def make_loader(d_in: int, n: int = 512, batch_size: int = 64) -> DataLoader:
    x = torch.randn(n, d_in)
    y = torch.sin(x @ torch.randn(d_in, 1))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)

    logger = DLTLogger(
        "lightning-demo",
        save_dir=args.runs_dir,
        config={"epochs": args.epochs, "batch_size": 64},
        save_every=args.save_every,
        log_git=True,
        async_save=True,  # YAML serialisation happens off the training thread
        jsonl_metrics=True,  # optional append-only metric stream for long runs
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=10,
        enable_progress_bar=True,
        # ---- the actual accelerations (uncomment on a GPU box) ---------- #
        # accelerator="gpu",
        # devices=4,
        # strategy="ddp",                 # or "fsdp" for large models
        # precision="16-mixed",           # or "bf16-mixed" on Ampere+
        # accumulate_grad_batches=2,
        # # torch>=2.0:
        # compile=True,
    )
    trainer.fit(MLP(), train_dataloaders=make_loader(16))

    print(f"run directory: {Path(logger.save_dir).parent}")
    print(f"yaml log:      {logger._logger.path}")
    return logger._logger.path


if __name__ == "__main__":
    main()
