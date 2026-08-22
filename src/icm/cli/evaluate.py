"""``icm-eval`` - evaluate a trained policy, or the scripted expert, in the environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="icm-eval", description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint; omit to evaluate the expert")
    ap.add_argument("-n", "--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--state-key", default="privileged", choices=["proprio", "privileged"])
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--image-size", type=int, default=84)
    ap.add_argument("--gif", default=None, help="write a GIF of the first episode here")
    ap.add_argument("--gif-camera", default="scene")
    ap.add_argument("--json", default=None, help="write results as JSON")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from ..envs.pick_place import EnvConfig, PickPlaceEnv
    from ..eval.metrics import evaluate_agent
    from ..eval.rollout import ScriptedAgent, rollout

    need_images = bool(args.images) or args.gif
    cameras = tuple({k.rsplit("_", 1)[0] for k in args.images} | ({args.gif_camera} if args.gif else set()))
    env = PickPlaceEnv(
        EnvConfig(render_images=bool(args.images), cameras=cameras or ("wrist", "scene"),
                  use_depth=False, image_size=args.image_size),
        seed=args.seed,
    )

    if args.checkpoint:
        from ..policies.runner import PolicyAgent, RunnerConfig
        from ..training.train_bc import load_policy

        policy = load_policy(args.checkpoint, device=args.device)
        agent = PolicyAgent(policy, RunnerConfig(state_key=args.state_key, device=args.device))
        label = f"policy {Path(args.checkpoint).parent.name}"
    else:
        agent = ScriptedAgent()
        label = "scripted expert"

    result = evaluate_agent(env, agent, n_episodes=args.episodes, seed=args.seed)
    if not args.quiet:
        print(f"{label}: {result}")
        breakdown = result.failure_breakdown()
        if breakdown:
            print(f"  failures by final phase: {breakdown}")
        print(f"  mean episode length: {result.to_dict()['mean_episode_length']:.1f}")

    if args.gif:
        frames = rollout(env, agent, seed=args.seed, render_camera=args.gif_camera).frames
        _write_gif(frames, args.gif)
        if not args.quiet:
            print(f"  wrote {args.gif} ({len(frames)} frames)")

    payload = {"label": label, **result.to_dict(), "failures": result.failure_breakdown()}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, default=float))
    env.close()
    return 0


def _write_gif(frames, path, fps: int = 20) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("GIF export needs imageio: pip install 'icm[viz]'") from exc
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
