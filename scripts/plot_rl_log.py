"""Parse an ``examples/rl_ray_ppo.py`` run log and plot the training curve.

Usage::

    python scripts/plot_rl_log.py logs/rl_impala_XXXX.log -o reports/rl_reward.png

The figure is written with :func:`dlt.plot_utils.packplot`, so the PNG carries
the plotting code and the data needed to re-render it::

    from dlt import retrieve_data, retrieve_plot
    retrieve_data("reports/rl_reward.png")
    retrieve_plot("reports/rl_reward.png", allow_execute=True)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from dlt.plot_utils import packplot

LINE = re.compile(r"^\[(\d+)\] avg_reward=([-\d.eE+nan]+) loss=([-\d.eE+nan]+)", re.M)


def parse_log(path: str | Path) -> dict[str, np.ndarray]:
    text = Path(path).read_text(errors="replace")
    rows = LINE.findall(text)
    if not rows:
        raise ValueError(f"no '[...] avg_reward=... loss=...' lines found in {path}")
    return {
        "iterations": np.array([int(r[0]) for r in rows], dtype=float),
        "reward": np.array([float(r[1]) for r in rows]),
        "loss": np.array([float(r[2]) for r in rows]),
    }


def moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    window = max(1, min(window, values.size))
    return np.convolve(values, np.ones(window) / window, mode="valid")


@packplot
def plot_rl_curve(iterations, reward, loss, filename, title="IMPALA V-trace + PopArt"):
    """Reward (left) and loss (right) with a moving average overlay."""
    import matplotlib.pyplot as plt

    window = max(1, min(25, reward.size))
    fig, (ax_reward, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))

    ax_reward.plot(iterations, reward, alpha=0.25, color="tab:blue", label="raw")
    ax_reward.plot(
        iterations[window - 1 :], moving_average(reward, window), color="tab:blue",
        label=f"moving avg ({window})",
    )
    ax_reward.set_xlabel("iteration")
    ax_reward.set_ylabel("avg per-step reward")
    ax_reward.set_title(title)
    ax_reward.grid(alpha=0.3)
    ax_reward.legend()

    ax_loss.plot(iterations, loss, alpha=0.25, color="tab:orange")
    ax_loss.plot(iterations[window - 1 :], moving_average(loss, window), color="tab:orange")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("training loss")
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="path to the run log")
    parser.add_argument("-o", "--output", default="reports/rl_reward.png")
    parser.add_argument("--title", default="IMPALA V-trace + PopArt")
    args = parser.parse_args(argv)

    data = parse_log(args.log)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_rl_curve(data["iterations"], data["reward"], data["loss"], output, title=args.title)
    print(
        f"wrote {output} | {data['iterations'].size} iterations | "
        f"reward first100={data['reward'][:100].mean():+.2f} "
        f"last100={data['reward'][-100:].mean():+.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
