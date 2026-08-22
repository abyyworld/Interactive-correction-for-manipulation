"""The attribution study: does the supervisor blame the phase that caused the error?

Protocol
--------
For each episode: inject one fault at a known phase, run the scripted agent under
a supervisor that can only see what is on screen, and record where the supervisor
took over and where they said the error was. Compare both against the truth.

Two contrasts carry the result:

Three attributions are measured, and keeping them apart is the point:

``onset``
    The phase the robot is in when the supervisor actually takes the controls.
    This is what naive HG-DAgger credits, since it relabels from the takeover
    timestamp onward. It is confounded in an interesting way: after a dropped
    object the phase tracker correctly reverts to APPROACH, so by the time a
    human has reacted the robot no longer looks like it is in the phase where
    the failure appeared.
``symptom``
    The phase at the instant the failure first became visible. This is the best
    an ideal observer could do from the timestamp alone.
``stated``
    What the supervisor says caused it, which is the only one that can point
    *backwards* past the symptom to the cause.

1. **Delayed vs immediate faults.** If misattribution were an artefact of the
   measurement, faults whose symptom appears in the same phase as their cause
   would be misattributed just as often as faults whose symptom is a phase or
   two later. They should not be.
2. **weak_grip vs lift_slip.** Both end with the object falling during the lift,
   so the symptom looks identical; only the true cause differs. Any gap between
   them cannot be explained by the failures looking different.

Everything is written through ``interventionkit``, so the analysis path is the
same one an external user of the tool would take.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from interventionkit import InterventionRecorder, RunReader, analyse
from interventionkit.attribution import per_phase_breakdown

from ..control.scripted import ExpertConfig, ScriptedExpert
from ..envs.faults import FAULT_PHASES, FaultInjector, FaultSpec, FaultType
from ..envs.pick_place import EnvConfig, PickPlaceEnv
from ..envs.phases import Phase
from ..eval.rollout import ScriptedAgent, rollout
from .supervisor import SupervisorConfig, SyntheticSupervisor

PHASE_NAMES = ("approach", "grasp", "lift", "place")

DEFAULT_FAULTS: tuple[FaultType, ...] = (
    FaultType.GRASP_OFFSET,
    FaultType.PREMATURE_CLOSE,
    FaultType.WEAK_GRIP,
    FaultType.LIFT_SLIP,
    FaultType.EARLY_RELEASE,
    FaultType.WRONG_OBJECT,
)


@dataclass
class StudyConfig:
    episodes_per_fault: int = 40
    faults: tuple[FaultType, ...] = DEFAULT_FAULTS
    #: Include fault-free episodes to measure the unnecessary-intervention rate.
    control_episodes: int = 40
    severity: float = 0.9
    trace_accuracy: float = 0.35
    seed: int = 0
    record_keys: tuple[str, ...] = ("proprio",)
    render_images: bool = False
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["faults"] = [f.value for f in self.faults]
        return d


def run_attribution_study(
    out_dir: str | Path,
    config: StudyConfig | None = None,
    env: PickPlaceEnv | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the study, write an interventionkit run, and return the summary."""
    cfg = config or StudyConfig()
    out_dir = Path(out_dir)
    owns_env = env is None
    if env is None:
        env = PickPlaceEnv(EnvConfig(render_images=cfg.render_images), seed=cfg.seed)

    sup_cfg = SupervisorConfig(**{**asdict(cfg.supervisor), "trace_accuracy": cfg.trace_accuracy})
    supervisor = SyntheticSupervisor(sup_cfg, cfg.expert, rng=np.random.default_rng(cfg.seed + 1))

    recorder = InterventionRecorder(
        out_dir,
        task="pick_place",
        phase_names=PHASE_NAMES,
        config=cfg.to_dict(),
        notes="synthetic supervisor; not human data",
    )

    jobs: list[tuple[FaultType | None, int]] = []
    for ft in cfg.faults:
        jobs += [(ft, i) for i in range(cfg.episodes_per_fault)]
    jobs += [(None, i) for i in range(cfg.control_episodes)]

    per_fault: dict[str, dict[str, Any]] = {}
    for idx, (ft, i) in enumerate(jobs):
        seed = cfg.seed * 100_000 + idx
        if ft is None:
            agent = ScriptedAgent(ScriptedExpert(cfg.expert))
        else:
            injector = FaultInjector(
                FaultSpec(type=ft, severity=cfg.severity), np.random.default_rng(seed)
            )
            agent = ScriptedAgent(ScriptedExpert(cfg.expert), injector)

        with recorder.episode(seed=seed, instruction=env.default_instruction()) as ep:
            result = rollout(
                env, agent, supervisor=supervisor, recorder=ep, seed=seed,
                record_keys=cfg.record_keys, supervisor_actor="expert",
            )

        key = ft.value if ft is not None else "none"
        bucket = per_fault.setdefault(
            key,
            {"n": 0, "detected": 0, "success": 0, "takeover_steps": [], "attributions": []},
        )
        bucket["n"] += 1
        bucket["detected"] += int(result.intervened)
        bucket["success"] += int(result.success)
        if result.takeover_step is not None:
            bucket["takeover_steps"].append(result.takeover_step)
        if result.attribution is not None:
            bucket["attributions"].append(result.attribution)

        if progress and (idx + 1) % 25 == 0:
            print(f"  [{idx + 1}/{len(jobs)}] episodes", flush=True)

    if owns_env:
        env.close()

    reader = RunReader(out_dir)
    episodes = reader.episodes()
    summary = analyse(episodes, n_phases=4, phase_names=PHASE_NAMES)

    by_lag = _group_by_lag(episodes)
    report = {
        "config": cfg.to_dict(),
        "run_stats": reader.stats(),
        "attribution": summary.to_dict(),
        "per_root_phase": per_phase_breakdown(episodes, n_phases=4),
        "by_lag_class": by_lag,
        "per_fault": {
            k: {
                "n": v["n"],
                "detection_rate": v["detected"] / v["n"],
                "success_rate_with_correction": v["success"] / v["n"],
                "mean_takeover_step": float(np.mean(v["takeover_steps"])) if v["takeover_steps"] else float("nan"),
            }
            for k, v in per_fault.items()
        },
        "unnecessary_intervention_rate": (
            per_fault["none"]["detected"] / per_fault["none"]["n"] if "none" in per_fault else float("nan")
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def _group_by_lag(episodes) -> dict[str, dict[str, float]]:
    """Misattribution grouped by how far the symptom is from the cause.

    This is the comparison the study exists to make, so it is computed here
    rather than left for a reader to assemble from the per-fault table.
    """
    lag_of = {ft.value: FAULT_PHASES[ft][2] for ft in FaultType}
    buckets: dict[str, dict[str, list]] = {}
    for ep in episodes:
        gt = ep.ground_truth or {}
        fault = gt.get("fault")
        if not fault or fault == "none" or not ep.interventions or gt.get("root_phase") is None:
            continue
        lag = lag_of.get(fault, "unknown")
        b = buckets.setdefault(lag, {"onset": [], "symptom": [], "stated": [], "delay": []})
        seg = ep.interventions[0]
        root = int(gt["root_phase"])
        b["onset"].append(int(seg.onset_phase != root))
        if gt.get("symptom_phase") is not None:
            b["symptom"].append(int(int(gt["symptom_phase"]) != root))
        if seg.attributed_phase is not None:
            b["stated"].append(int(int(seg.attributed_phase) != root))
        if gt.get("root_onset_step") is not None:
            b["delay"].append(seg.start - int(gt["root_onset_step"]))
    return {
        lag: {
            "n": float(len(v["onset"])),
            "onset_misattribution_rate": float(np.mean(v["onset"])) if v["onset"] else float("nan"),
            "symptom_misattribution_rate": float(np.mean(v["symptom"])) if v["symptom"] else float("nan"),
            "stated_misattribution_rate": float(np.mean(v["stated"])) if v["stated"] else float("nan"),
            "mean_detection_lag": float(np.mean(v["delay"])) if v["delay"] else float("nan"),
        }
        for lag, v in sorted(buckets.items())
    }


def sweep_trace_accuracy(
    out_root: str | Path,
    accuracies: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    config: StudyConfig | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Repeat the study across supervisor tracing ability.

    The point is not any single value. Real tracing accuracy is unknown, so the
    deliverable is the whole curve: when the VR study returns a number, the
    corresponding degradation can be read straight off it.
    """
    out_root = Path(out_root)
    base = config or StudyConfig()
    env = PickPlaceEnv(EnvConfig(render_images=base.render_images), seed=base.seed)
    results = {}
    for acc in accuracies:
        cfg = StudyConfig(**{**asdict(base), "trace_accuracy": acc,
                             "faults": base.faults,
                             "supervisor": base.supervisor, "expert": base.expert})
        if progress:
            print(f"[sweep] trace_accuracy={acc}", flush=True)
        results[str(acc)] = run_attribution_study(
            out_root / f"trace_{acc:.2f}", cfg, env=env, progress=False
        )
    env.close()
    (out_root / "sweep.json").write_text(json.dumps(results, indent=2, default=float))
    return results
