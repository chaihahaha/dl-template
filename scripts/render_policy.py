"""Render a trained Pendulum policy from a checkpoint to an MP4.

The training example saves ``best.pt`` (weights + observation normaliser) when
run with ``--checkpoint-dir``::

    python examples/rl_ray_ppo.py --iterations 2000 ... --checkpoint-dir runs/impala-pendulum
    python scripts/render_policy.py runs/impala-pendulum/best.pt -o reports/pendulum_best.mp4

Use ``--random`` to render an untrained policy for comparison, and
``--episodes N`` to concatenate several episodes.

Requirements: ``gymnasium``, ``pygame`` (env rendering), ``imageio`` +
``imageio-ffmpeg`` (MP4 encoding).  Rendering is headless (SDL dummy driver), so
it works over SSH without a display.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_example():
    spec = importlib.util.spec_from_file_location(
        "rl_ray_ppo_example", ROOT / "examples" / "rl_ray_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def register_unpickle_aliases(example) -> None:
    """Checkpoints store objects defined in the training script's ``__main__``.

    When the training script ran as ``python examples/rl_ray_ppo.py`` its
    classes carry ``__module__ == "__main__"``, so unpickling here needs those
    names to resolve on *our* ``__main__`` too.
    """
    main = sys.modules.get("__main__")
    if main is None:
        return
    for name in ("AgentModel", "PopArtValue", "Normalizer", "ReplayMemory", "Context"):
        if not hasattr(main, name) and hasattr(example, name):
            setattr(main, name, getattr(example, name))


def build_policy(example, checkpoint, d_obs, d_act, low, high, random: bool):
    if random:
        model = example.AgentModel(8, d_obs, d_act, low, high)
        normalizer = example.Normalizer((d_obs,), np.float32)
        return model.eval(), normalizer, {"iteration": 0, "eval_return": float("nan")}

    register_unpickle_aliases(example)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = payload.get("params", {})
    model = example.AgentModel(
        int(params.get("ctx_size", 8)),
        d_obs,
        d_act,
        low,
        high,
        popart_beta=float(params.get("popart_beta", 0.99)),
    )
    model.load_state_dict(payload["state_dict"])
    normalizer = payload.get("normalizer")
    if normalizer is None:  # plain-stats checkpoints
        normalizer = example.Normalizer((d_obs,), np.float32)
        state = payload.get("normalizer_state")
        if state is not None:
            normalizer.load_state(state)
    meta = {
        "iteration": payload.get("iteration"),
        "eval_return": payload.get("eval_return"),
    }
    return model.eval(), normalizer, meta


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    height = 14 + 13 * len(lines)
    draw.rectangle([0, 0, image.width, height], fill=(0, 0, 0, 140))
    for index, text in enumerate(lines):
        draw.text((8, 4 + 13 * index), text, fill=(255, 255, 255, 255))
    return np.asarray(image)


@torch.no_grad()
def render_episode(env, model, normalizer, example, seed: int, max_steps: int = 200):
    obs, _ = env.reset(seed=seed)
    ctx_size = model.d1.in_features // obs.shape[0]
    context = [np.zeros_like(obs, dtype=np.float32) for _ in range(ctx_size)]
    context.append(normalizer.normalize_obs(obs).astype(np.float32))
    context.pop(0)

    frames, total = [], 0.0
    for step in range(max_steps):
        batch = torch.as_tensor(np.stack(context, 0), dtype=torch.float32).unsqueeze(0)
        distribution, _ = model.dist_value(batch)
        action = distribution.mean.numpy()[0]  # deterministic policy
        obs, reward, terminated, truncated, _ = env.step(action)
        context.append(normalizer.normalize_obs(obs).astype(np.float32))
        context.pop(0)
        total += float(reward)
        frames.append((env.render(), step, total))
        if terminated or truncated:
            break
    return frames, total


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", pixelformat="yuv420p",
        quality=8, macro_block_size=None,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", help="path to best.pt")
    parser.add_argument("-o", "--output", default="reports/pendulum_policy.mp4")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--random", action="store_true", help="render an untrained policy")
    args = parser.parse_args(argv)

    if not args.random and not args.checkpoint:
        parser.error("a checkpoint path is required unless --random is given")

    example = load_example()
    import gymnasium as gym

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    d_obs = env.observation_space.shape[0]
    d_act = env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high

    model, normalizer, meta = build_policy(example, args.checkpoint, d_obs, d_act, low, high, args.random)

    all_frames: list[np.ndarray] = []
    returns = []
    for episode in range(args.episodes):
        frames, episode_return = render_episode(
            env, model, normalizer, example, seed=args.seed + episode
        )
        returns.append(episode_return)
        header = [
            f"iteration {meta['iteration']}  eval_return {meta['eval_return']:.1f}"
            if meta["iteration"] is not None and np.isfinite(meta["eval_return"])
            else "untrained (random) policy",
            f"episode {episode + 1}/{args.episodes}   return {episode_return:.1f}",
        ]
        annotated = [
            annotate(frame, header + [f"step {step:3d}   running return {running:.1f}"])
            for frame, step, running in frames
        ]
        all_frames.extend(annotated)
        all_frames.extend([annotated[-1]] * (args.fps // 2))  # brief pause

    output = Path(args.output)
    write_video(all_frames, output, args.fps)
    print(
        f"wrote {output} | {len(all_frames)} frames | "
        f"episode returns: {[round(r, 1) for r in returns]} | mean {np.mean(returns):.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
