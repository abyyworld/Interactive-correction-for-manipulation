"""``icm-collect`` - generate demonstration episodes with the scripted expert.

Writes an interventionkit run that ``icm-train`` consumes directly. Demonstrations
are recorded as expert steps, so the same "supervision = corrections" rule that
selects human corrections also selects them, and the two mix without special
cases.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="icm-collect", description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", required=True, help="output run directory")
    ap.add_argument("-n", "--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images", action="store_true", help="render and store camera observations")
    ap.add_argument("--image-size", type=int, default=84)
    ap.add_argument("--cameras", nargs="*", default=["wrist", "scene"])
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--no-privileged", action="store_true",
                    help="omit ground-truth object poses from the recording")
    ap.add_argument("--only-successes", action="store_true",
                    help="discard failed episodes instead of recording them")
    ap.add_argument("--estimate-only", action="store_true",
                    help="print the predicted dataset size and exit")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from ..training.dataset import estimate_size

    if args.estimate_only or args.images:
        est = estimate_size(
            args.episodes, steps_per_episode=80, image_size=args.image_size,
            n_cameras=len(args.cameras) if args.images else 0, depth=args.depth,
        )
        msg = (f"[estimate] {args.episodes} episodes ~ {est['raw_gb']:.2f} GB raw / "
               f"{est['compressed_gb']:.2f} GB compressed ({est['frames']:,.0f} frames)")
        if not args.quiet:
            print(msg)
        if args.estimate_only:
            return 0
        if est["compressed_gb"] > 100:
            print("[estimate] refusing: over 100 GB. Lower --episodes, --image-size, or drop --depth.")
            return 2

    from interventionkit import InterventionRecorder

    from ..control.scripted import ScriptedExpert
    from ..envs.pick_place import EnvConfig, PickPlaceEnv
    from ..eval.rollout import ScriptedAgent, rollout

    env = PickPlaceEnv(
        EnvConfig(
            render_images=args.images,
            cameras=tuple(args.cameras),
            use_depth=args.depth,
            image_size=args.image_size,
            include_privileged=not args.no_privileged,
        ),
        seed=args.seed,
    )
    keys: list[str] = ["proprio"]
    if not args.no_privileged:
        keys.append("privileged")
    if args.images:
        for cam in args.cameras:
            keys.append(f"{cam}_rgb")
            if args.depth:
                keys.append(f"{cam}_depth")

    recorder = InterventionRecorder(
        args.out, task="pick_place",
        phase_names=("approach", "grasp", "lift", "place"),
        config={"source": "scripted_expert", "images": args.images, "seed": args.seed},
    )
    expert = ScriptedExpert()
    agent = ScriptedAgent(expert)

    n_success = 0
    t0 = time.time()
    for i in range(args.episodes):
        seed = args.seed * 100_000 + i
        if args.only_successes:
            # Peek first so a failed episode is never written at all.
            probe = rollout(env, agent, seed=seed)
            if not probe.success:
                continue
        with recorder.episode(seed=seed, instruction=env.default_instruction()) as ep:
            result = rollout(env, agent, recorder=ep, seed=seed, record_keys=tuple(keys),
                             record_agent_as="expert")
        n_success += int(result.success)
        if not args.quiet and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{args.episodes}] success {n_success}/{i+1} "
                  f"({(time.time()-t0)/(i+1):.2f} s/ep)", flush=True)

    env.close()
    summary = {"episodes": args.episodes, "successes": n_success,
               "seconds": time.time() - t0, "out": str(args.out)}
    Path(args.out, "collect.json").write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
