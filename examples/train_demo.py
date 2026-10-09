"""End-to-end demo of the dlt template.

Run it from the repository root::

    python examples/train_demo.py --epochs 30

It shows every tool working together:

* ``dlt.never_overwrite``  - checkpoints are backed up, not clobbered
* ``dlt.naming``           - a sortable, collision-free run directory
* ``dlt.yaml_logging``     - crash-safe YAML log with a ``latest:`` section
* ``dlt.plot_utils``       - the loss curve carries its own plotting code

Only ``torch`` (plus the optional plotting extras) is required; PyYAML is the
sole hard dependency of the library itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

# Importing this patches open/torch.save so a re-run can never destroy a
# previous checkpoint or log.
import dlt.never_overwrite as _never_overwrite  # noqa: F401
from dlt import ExperimentNamer, YamlLogger
from dlt.plot_utils import packplot


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@packplot
def plot_loss(history, filename, title="training loss"):
    """Plot the loss history and embed this code + the data in the PNG."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(history, label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--d-in", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="demo-mlp")
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    config = {k: v for k, v in vars(args).items()}
    run = ExperimentNamer(args.runs_dir, name=args.name).new_run(config)
    print(f"run directory: {run.run_dir}")

    # Synthetic regression task: y = sin(x @ w) so there is something to learn.
    w_true = torch.randn(args.d_in, 1)
    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(args.batch_size, args.d_in)
        y = torch.sin(x @ w_true)
        return x, y

    model = MLP(args.d_in, args.hidden, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    with YamlLogger(args.name, run=run, config=config, save_every=10, log_git=True) as log:
        with log.catch():
            history: list[float] = []
            for epoch in range(1, args.epochs + 1):
                epoch_loss = 0.0
                for _ in range(args.steps_per_epoch):
                    x, y = make_batch()
                    loss = loss_fn(model(x), y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach())
                mean_loss = epoch_loss / args.steps_per_epoch
                history.append(mean_loss)
                log.log(epoch=epoch, loss=mean_loss, lr=args.lr)

            # Checkpoint: a second run of this script backs it up instead of
            # silently destroying it (torch.save is wrapped by never_overwrite).
            ckpt = run.ckpts / "final.pt"
            torch.save({"model": model.state_dict(), "config": config}, ckpt)
            log.add_artifact(ckpt)

            figure = run.figures / "loss.png"
            try:
                plot_loss(history, figure, title=f"{args.name} loss")
                log.add_artifact(figure)
            except ImportError:
                log.add_info("matplotlib not installed; skipped the loss figure")

            log.add_info(f"finished {args.epochs} epochs")

    print(f"log:    {log.path}")
    print(f"latest: {log.latest}")
    return log.path


if __name__ == "__main__":
    main()
