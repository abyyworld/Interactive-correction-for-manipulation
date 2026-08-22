"""The shared rollout loop: one agent, one optional supervisor, one recorder.

Every experiment in this project is this loop with different pieces plugged in:

* expert only                  -> demonstration data
* faulty expert + supervisor   -> the attribution study
* learned policy + supervisor  -> HG-DAgger
* learned policy alone         -> evaluation

Writing it once matters more than it looks. The subtle part is the handover:
which actor a step is attributed to, when the fault stops being applied, and
when an attribution is captured. Getting that wrong in one of four near-copies
would produce a dataset that is mislabelled in only some conditions — the kind
of bug that survives right through to the results table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..control.scripted import ScriptedExpert
from ..envs.faults import FaultInjector
from ..envs.phases import Phase


class Agent(Protocol):
    """The thing being supervised."""

    def reset(self, env: Any, seed: int | None = None) -> None: ...

    def act(self, env: Any, obs: dict, info: dict, step: int) -> np.ndarray: ...


class ScriptedAgent:
    """The scripted expert, optionally with a fault injected. Also the corrective source."""

    def __init__(self, expert: ScriptedExpert | None = None, injector: FaultInjector | None = None):
        self.expert = expert or ScriptedExpert()
        self.injector = injector

    def reset(self, env, seed: int | None = None) -> None:
        self.expert.reset()
        if self.injector is not None:
            self.injector.reset(env, self.expert)

    def act(self, env, obs: dict, info: dict, step: int) -> np.ndarray:
        action = self.expert.act(env)
        if self.injector is not None:
            action = self.injector.modify(env, self.expert, action, step)
        return action

    def on_supervisor_engage(self, env) -> None:
        if self.injector is not None:
            self.injector.suspend(env)

    def ground_truth(self, env) -> dict:
        return self.injector.ground_truth(env) if self.injector is not None else {}


@dataclass
class RolloutResult:
    success: bool = False
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    final_phase: Phase = Phase.APPROACH

    intervened: bool = False
    takeover_step: int | None = None
    detect_reason: str = ""
    detect_phase: int | None = None
    detect_step: int | None = None
    attribution: int | None = None
    confidence: float | None = None

    phases: list[int] = field(default_factory=list)
    actors: list[str] = field(default_factory=list)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    frames: list[np.ndarray] = field(default_factory=list)
    episode_id: str = ""

    @property
    def n_corrected(self) -> int:
        return sum(1 for a in self.actors if a != "policy")

    @property
    def phase_visits(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.phases:
            name = Phase(p).label
            out[name] = out.get(name, 0) + 1
        return out


def rollout(
    env,
    agent: Agent,
    supervisor=None,
    recorder=None,
    *,
    seed: int | None = None,
    instruction: str | None = None,
    record_keys: tuple[str, ...] = ("proprio",),
    record_agent_as: str = "policy",
    max_steps: int | None = None,
    render_camera: str | None = None,
    render_every: int = 1,
    supervisor_actor: str = "expert",
) -> RolloutResult:
    """Run one episode.

    ``supervisor_actor`` distinguishes a simulated supervisor ("expert") from a
    real person ("human") in the recorded data. Both count as interventions, but
    conflating them would let synthetic results be reported as human ones.

    ``record_agent_as`` labels the agent's own steps. It is "policy" during
    interactive collection, and "expert" when the agent *is* the scripted expert
    generating demonstrations - otherwise the supervision rule that selects
    corrective frames would discard every demonstration.
    """
    obs = env.reset(seed=seed, instruction=instruction)
    info = env.info_dict()
    agent.reset(env, seed=seed)

    engage_hook_installed = False
    if supervisor is not None:
        root_phase = None
        gt_probe = getattr(agent, "ground_truth", None)
        if gt_probe is not None:
            gt = gt_probe(env)
            if gt.get("root_phase") is not None and gt.get("fault") not in (None, "none"):
                root_phase = Phase(int(gt["root_phase"]))
        if hasattr(supervisor, "reset"):
            try:
                supervisor.reset(env, seed=seed, true_root_phase=root_phase)
            except TypeError:  # a device that does not model ground truth
                supervisor.reset(env, seed=seed)
        if hasattr(supervisor, "set_engage_callback") and hasattr(agent, "on_supervisor_engage"):
            supervisor.set_engage_callback(lambda: agent.on_supervisor_engage(env))
            engage_hook_installed = True

    result = RolloutResult(episode_id=getattr(recorder, "_w", None) and recorder._w.episode_id or "")
    limit = max_steps or env.config.max_episode_steps
    was_engaged = False

    for step in range(limit):
        command = None
        if supervisor is not None:
            command = supervisor.poll(env, obs, info, step)

        if command is not None and command.abort:
            break

        if command is not None and command.engaged:
            action = np.asarray(command.action, dtype=float)
            actor = supervisor_actor
            if not was_engaged:
                was_engaged = True
                result.intervened = True
                result.takeover_step = step
                result.detect_reason = str(command.extra.get("detect_reason", ""))
                dp = command.extra.get("detect_phase", -1)
                ds = command.extra.get("detect_step", -1)
                result.detect_phase = int(dp) if dp is not None and int(dp) >= 0 else None
                result.detect_step = int(ds) if ds is not None and int(ds) >= 0 else None
                # Fall back to suspending the fault here if the supervisor does
                # not support the engage callback (e.g. a real teleop device).
                if not engage_hook_installed and hasattr(agent, "on_supervisor_engage"):
                    agent.on_supervisor_engage(env)
            if command.attribution is not None and result.attribution is None:
                result.attribution = int(command.attribution)
                result.confidence = command.confidence
        else:
            action = np.asarray(agent.act(env, obs, info, step), dtype=float)
            actor = record_agent_as

        if recorder is not None:
            payload = {k: obs[k] for k in record_keys if k in obs}
            method = recorder.policy_step if actor == "policy" else (
                recorder.human_step if actor == "human" else recorder.expert_step
            )
            method(action, phase=int(info["phase"]), **payload)
            if command is not None and command.attribution is not None and recorder.intervened:
                already = recorder.interventions[-1].attributed_phase is not None
                if not already:
                    recorder.attribute(
                        int(command.attribution),
                        confidence=command.confidence,
                        notes=command.notes,
                    )

        result.actors.append(actor)
        result.phases.append(int(info["phase"]))

        if render_camera and step % render_every == 0:
            result.frames.append(env.render_frame(render_camera))

        obs, reward, terminated, truncated, info = env.step(action)
        result.steps = step + 1
        if terminated or truncated:
            result.terminated, result.truncated = terminated, truncated
            break

    result.success = bool(info["success"])
    result.final_phase = Phase(int(info["phase"]))
    gt_fn = getattr(agent, "ground_truth", None)
    result.ground_truth = gt_fn(env) if gt_fn is not None else {}
    if result.takeover_step is not None:
        result.ground_truth["takeover_step"] = result.takeover_step
        result.ground_truth["detect_reason"] = result.detect_reason
        result.ground_truth["symptom_phase"] = result.detect_phase
        result.ground_truth["symptom_step"] = result.detect_step

    if recorder is not None:
        # The phase timeline is what lets a credit-assignment strategy rewind to
        # "the start of the phase the supervisor blamed" after the fact.
        recorder.finish(
            success=result.success,
            ground_truth=result.ground_truth,
            extra={"phase_timeline": [int(p) for p in result.phases]},
        )
    return result
