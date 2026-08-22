"""``icm-teleop`` - drive the robot by hand and record the corrections."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="icm-teleop", description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", required=True, help="output run directory")
    ap.add_argument("--device", default="keyboard", choices=["keyboard", "vr", "synthetic"])
    ap.add_argument("-n", "--episodes", type=int, default=10)
    ap.add_argument("--port", type=int, default=5555, help="UDP port for the VR client")
    ap.add_argument("--position-scale", type=float, default=2.5,
                    help="robot metres per metre of hand movement (VR only)")
    ap.add_argument("--agent", default="expert", choices=["expert", "faulty", "checkpoint"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--fault", default="grasp_offset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import numpy as np
    from interventionkit import InterventionRecorder

    from ..control.scripted import ScriptedExpert
    from ..envs.faults import FaultInjector, FaultSpec, FaultType
    from ..envs.pick_place import EnvConfig, PickPlaceEnv
    from ..eval.rollout import ScriptedAgent, rollout

    env = PickPlaceEnv(EnvConfig(render_images=args.images), seed=args.seed)

    if args.device == "keyboard":
        from ..teleop.keyboard import KeyboardTeleop

        supervisor = KeyboardTeleop()
        actor = "human"
    elif args.device == "vr":
        from ..teleop.vr import VRConfig, VRTeleop

        supervisor = VRTeleop(VRConfig(port=args.port, position_scale=args.position_scale))
        print(f"[teleop] listening for VR packets on UDP :{args.port}")
        actor = "human"
    else:
        from ..study.supervisor import SyntheticSupervisor

        supervisor = SyntheticSupervisor()
        actor = "expert"

    recorder = InterventionRecorder(
        args.out, task="pick_place", phase_names=("approach", "grasp", "lift", "place"),
        config={"device": args.device, "agent": args.agent},
    )

    keys = ["proprio", "privileged"]
    if args.images:
        keys += ["wrist_rgb", "scene_rgb"]

    for i in range(args.episodes):
        seed = args.seed * 1000 + i
        if args.agent == "faulty":
            rng = np.random.default_rng(seed)
            agent = ScriptedAgent(
                ScriptedExpert(), FaultInjector(FaultSpec(type=FaultType(args.fault), severity=0.9), rng)
            )
        elif args.agent == "checkpoint":
            from ..policies.runner import PolicyAgent, RunnerConfig
            from ..training.train_bc import load_policy

            agent = PolicyAgent(load_policy(args.checkpoint), RunnerConfig(state_key="privileged"))
        else:
            agent = ScriptedAgent()

        with recorder.episode(seed=seed, instruction=env.default_instruction()) as ep:
            result = rollout(env, agent, supervisor=supervisor, recorder=ep, seed=seed,
                             record_keys=tuple(keys), supervisor_actor=actor)
        print(f"episode {i}: success={result.success} intervened={result.intervened} "
              f"takeover={result.takeover_step} blamed={result.attribution}")
        if result.ground_truth.get("aborted"):
            break

    supervisor.close()
    env.close()
    print(f"\nwrote {args.episodes} episodes to {args.out}")
    print(f"analyse with:  ik-report {args.out} -o {args.out}/report.html")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
