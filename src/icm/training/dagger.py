"""Human-gated DAgger with rewind-and-redemonstrate.

The protocol
------------
Standard HG-DAgger relabels from the moment the supervisor takes the controls.
That is fine when the mistake *is* where they took over, and wrong whenever the
cause was earlier: the causal states never receive a corrective action, so the
policy keeps producing the behaviour that led to the failure and only learns to
tidy up afterwards.

This module implements the alternative that attribution makes possible. When the
supervisor intervenes and names a phase, the episode is **rewound** to the start
of that phase using an exact state snapshot, and the correction is demonstrated
from there. The recorded episode is then policy actions up to the rewind point
and corrective actions after it.

That makes the experiment direct. The rewind target is the only thing that
changes between conditions:

===========  ==================================================================
 ONSET        rewind to the takeover step (i.e. no rewind) - today's HG-DAgger
 SYMPTOM      rewind to where the failure first became visible
 STATED       rewind to the start of the phase the supervisor blamed
 ORACLE       rewind to the true root cause - not implementable, the ceiling
===========  ==================================================================

If the supervisor's attribution is wrong, STATED rewinds to a phase that was
already fine and the corrective data misses the causal states entirely. The gap
between STATED and ORACLE, at a given tracing accuracy, is the cost of
misattribution measured in task success.

Exact snapshot/restore is what makes this possible at all, which is why the
environment guarantees it bit-exactly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from interventionkit import InterventionRecorder

from ..control.scripted import ExpertConfig, ScriptedExpert
from ..envs.phases import Phase
from ..envs.pick_place import PickPlaceEnv
from ..eval.rollout import ScriptedAgent
from ..study.supervisor import SyntheticSupervisor
from .weighting import CreditAssignment

PHASE_NAMES = ("approach", "grasp", "lift", "place")


@dataclass
class InteractiveEpisodeResult:
    success_before: bool = False
    success_after: bool = False
    intervened: bool = False
    takeover_step: int | None = None
    symptom_step: int | None = None
    rewind_step: int | None = None
    attributed_phase: int | None = None
    true_root_phase: int | None = None
    root_onset_step: int | None = None
    n_policy_steps: int = 0
    n_correction_steps: int = 0
    detect_reason: str = ""


def first_step_of_phase(phases: list[int], phase: int) -> int | None:
    for t, p in enumerate(phases):
        if int(p) == int(phase):
            return t
    return None


def resolve_rewind_step(
    strategy: CreditAssignment,
    *,
    takeover_step: int,
    symptom_step: int | None,
    attributed_phase: int | None,
    root_onset_step: int | None,
    phases: list[int],
) -> int:
    """Where to rewind to, given the strategy and what the supervisor said."""
    if strategy is CreditAssignment.ONSET:
        return takeover_step
    if strategy is CreditAssignment.SYMPTOM:
        return symptom_step if symptom_step is not None else takeover_step
    if strategy is CreditAssignment.STATED:
        if attributed_phase is None:
            return takeover_step
        step = first_step_of_phase(phases, attributed_phase)
        return step if step is not None else takeover_step
    if strategy is CreditAssignment.ORACLE:
        return root_onset_step if root_onset_step is not None else takeover_step
    raise ValueError(strategy)


def collect_interactive_episode(
    env: PickPlaceEnv,
    agent,
    supervisor: SyntheticSupervisor,
    recorder_episode,
    *,
    seed: int,
    strategy: CreditAssignment,
    corrective_expert: ScriptedExpert | None = None,
    record_keys: tuple[str, ...] = ("proprio", "privileged"),
    snapshot_every: int = 1,
) -> InteractiveEpisodeResult:
    """Run one supervised episode, then rewind and re-demonstrate the correction.

    Snapshots cost roughly 500 bytes per step (qpos, qvel, ctrl and the
    controller setpoint), so keeping one per step for a single episode is about
    100 kB - negligible, and it removes any quantisation in the rewind target.
    """
    res = InteractiveEpisodeResult()
    corrective = corrective_expert or ScriptedExpert()

    obs = env.reset(seed=seed)
    info = env.info_dict()
    agent.reset(env, seed=seed)

    root_phase = None
    gt_fn = getattr(agent, "ground_truth", None)
    if gt_fn is not None:
        gt = gt_fn(env)
        if gt.get("fault") not in (None, "none") and gt.get("root_phase") is not None:
            root_phase = Phase(int(gt["root_phase"]))
            res.true_root_phase = int(gt["root_phase"])
    try:
        supervisor.reset(env, seed=seed, true_root_phase=root_phase)
    except TypeError:
        supervisor.reset(env, seed=seed)

    snapshots: dict[int, dict] = {}
    phases: list[int] = []
    frames: list[tuple[np.ndarray, dict, int]] = []  # (action, obs, phase)

    takeover = None
    for step in range(env.config.max_episode_steps):
        if step % snapshot_every == 0:
            snapshots[step] = env.get_state()
        phases.append(int(info["phase"]))

        command = supervisor.poll(env, obs, info, step)
        if command.engaged:
            takeover = step
            res.detect_reason = str(command.extra.get("detect_reason", ""))
            sym = command.extra.get("detect_phase", -1)
            res.symptom_step = command.extra.get("detect_step", None)
            if res.symptom_step is not None and int(res.symptom_step) < 0:
                res.symptom_step = None
            if command.attribution is not None:
                res.attributed_phase = int(command.attribution)
            break

        action = np.asarray(agent.act(env, obs, info, step), dtype=float)
        frames.append((action, {k: obs[k] for k in record_keys if k in obs}, int(info["phase"])))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    res.success_before = bool(info["success"])
    if gt_fn is not None:
        gt = gt_fn(env)
        onset = gt.get("root_onset_step")
        res.root_onset_step = int(onset) if onset is not None else None

    if takeover is None:
        # No intervention: record the episode as the policy produced it.
        for action, payload, phase in frames:
            recorder_episode.policy_step(action, phase=phase, **payload)
        res.n_policy_steps = len(frames)
        res.success_after = res.success_before
        recorder_episode.finish(
            success=res.success_before,
            ground_truth=(gt_fn(env) if gt_fn is not None else {}),
            extra={"phase_timeline": phases, "strategy": strategy.value},
        )
        return res

    res.intervened = True
    res.takeover_step = takeover

    rewind = resolve_rewind_step(
        strategy,
        takeover_step=takeover,
        symptom_step=res.symptom_step,
        attributed_phase=res.attributed_phase,
        root_onset_step=res.root_onset_step,
        phases=phases,
    )
    rewind = int(np.clip(rewind, 0, max(0, len(frames))))
    res.rewind_step = rewind

    # Keep the policy's own steps up to the rewind point; they are context, not
    # supervision, and the dataset excludes them from the training signal.
    for action, payload, phase in frames[:rewind]:
        recorder_episode.policy_step(action, phase=phase, **payload)
    res.n_policy_steps = min(rewind, len(frames))

    # Rewind and re-demonstrate.
    snap_key = min(snapshots.keys(), key=lambda k: abs(k - rewind)) if snapshots else None
    if snap_key is not None:
        env.set_state(snapshots[snap_key])
    if hasattr(agent, "on_supervisor_engage"):
        agent.on_supervisor_engage(env)  # the fault is the agent's, not the robot's

    obs = env.observation()
    info = env.info_dict()
    corrective.resume_from_state(env)
    remaining = env.config.max_episode_steps - res.n_policy_steps
    for step in range(max(0, remaining)):
        action = corrective.act(env)
        payload = {k: obs[k] for k in record_keys if k in obs}
        recorder_episode.expert_step(action, phase=int(info["phase"]), **payload)
        res.n_correction_steps += 1
        obs, reward, terminated, truncated, info = env.step(action)
        phases.append(int(info["phase"]))
        if terminated or truncated:
            break

    res.success_after = bool(info["success"])
    if res.attributed_phase is not None:
        recorder_episode.attribute(res.attributed_phase)

    ground_truth = gt_fn(env) if gt_fn is not None else {}
    ground_truth.update(
        {
            "takeover_step": takeover,
            "symptom_step": res.symptom_step,
            "rewind_step": rewind,
            "strategy": strategy.value,
            "detect_reason": res.detect_reason,
        }
    )
    recorder_episode.finish(
        success=res.success_after,
        ground_truth=ground_truth,
        extra={"phase_timeline": phases, "strategy": strategy.value},
    )
    return res


@dataclass
class CollectionSummary:
    episodes: int = 0
    intervened: int = 0
    success_before: int = 0
    success_after: int = 0
    correction_steps: int = 0
    rewind_offsets: list[int] = field(default_factory=list)
    attribution_correct: int = 0
    attribution_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        n = max(self.episodes, 1)
        return {
            "episodes": self.episodes,
            "intervention_rate": self.intervened / n,
            "success_before_correction": self.success_before / n,
            "success_after_correction": self.success_after / n,
            "correction_steps": self.correction_steps,
            "mean_rewind_offset": float(np.mean(self.rewind_offsets)) if self.rewind_offsets else 0.0,
            "attribution_accuracy": (
                self.attribution_correct / self.attribution_total if self.attribution_total else float("nan")
            ),
        }


def collect_round(
    env: PickPlaceEnv,
    make_agent,
    supervisor: SyntheticSupervisor,
    out_dir: str | Path,
    *,
    n_episodes: int,
    strategy: CreditAssignment,
    seed: int = 0,
    record_keys: tuple[str, ...] = ("proprio", "privileged"),
    expert_config: ExpertConfig | None = None,
    progress: bool = False,
) -> tuple[Path, CollectionSummary]:
    """Collect one round of supervised episodes under a credit-assignment strategy."""
    out_dir = Path(out_dir)
    recorder = InterventionRecorder(
        out_dir, task="pick_place", phase_names=PHASE_NAMES,
        config={"strategy": strategy.value, "n_episodes": n_episodes},
    )
    corrective = ScriptedExpert(expert_config or ExpertConfig())
    summary = CollectionSummary()

    for i in range(n_episodes):
        ep_seed = seed * 10_000 + i
        agent = make_agent(i)
        with recorder.episode(seed=ep_seed, instruction=env.default_instruction()) as ep:
            r = collect_interactive_episode(
                env, agent, supervisor, ep, seed=ep_seed, strategy=strategy,
                corrective_expert=corrective, record_keys=record_keys,
            )
        summary.episodes += 1
        summary.intervened += int(r.intervened)
        summary.success_before += int(r.success_before)
        summary.success_after += int(r.success_after)
        summary.correction_steps += r.n_correction_steps
        if r.intervened and r.takeover_step is not None and r.rewind_step is not None:
            summary.rewind_offsets.append(r.takeover_step - r.rewind_step)
        if r.attributed_phase is not None and r.true_root_phase is not None:
            summary.attribution_total += 1
            summary.attribution_correct += int(r.attributed_phase == r.true_root_phase)
        if progress and (i + 1) % 25 == 0:
            print(f"    [{i+1}/{n_episodes}]", flush=True)

    (out_dir / "collection.json").write_text(json.dumps(summary.to_dict(), indent=2, default=float))
    return out_dir, summary
