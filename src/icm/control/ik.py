"""Damped least-squares inverse kinematics for the Panda arm.

Why DLS and not an analytic solver
----------------------------------
The Panda is 7-DoF: redundant for a 6-DoF Cartesian target, so there is no unique
analytic solution and closed-form solvers must pick an arbitrary redundancy
parameter. Damped least squares handles the redundancy naturally, degrades
gracefully near singularities (which a pseudo-inverse does not — it explodes),
and lets us push the redundant DoF toward a comfortable posture through the
null space. It is also ~30 lines, so it is easy to reason about when a grasp
goes wrong.

The damping term is the important part. Near a singularity ``J Jᵀ`` becomes
ill-conditioned and an undamped pseudo-inverse produces enormous joint
velocities. Adding ``λ²I`` bounds the solution at the cost of a small tracking
error, which for manipulation is always the right trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    qpos: np.ndarray  # arm joint positions (7,)
    pos_err: float  # final position error, metres
    rot_err: float  # final orientation error, radians
    iterations: int
    converged: bool


def quat_error(target_quat: np.ndarray, current_mat: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``current`` orientation to ``target``.

    Frame matters here and getting it wrong is subtle. ``mju_subQuat`` returns the
    difference expressed in the *local* frame of its second argument, whereas
    ``mj_jacSite`` produces an angular Jacobian in the *world* frame. Mixing the
    two yields an error term the Jacobian cannot act on, and the solver stalls at
    a ~pi orientation error that looks like an unreachable target rather than a
    bug. Composing ``target * conj(current)`` and converting to a velocity keeps
    everything in world coordinates.
    """
    current_quat = np.zeros(4)
    mujoco.mju_mat2Quat(current_quat, np.ascontiguousarray(current_mat).flatten())
    conj = np.zeros(4)
    mujoco.mju_negQuat(conj, current_quat)
    delta = np.zeros(4)
    mujoco.mju_mulQuat(delta, target_quat, conj)
    err = np.zeros(3)
    mujoco.mju_quat2Vel(err, delta, 1.0)
    return err


def solve_ik(
    model,
    data,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None = None,
    *,
    dof_indices: np.ndarray | None = None,
    qpos_indices: np.ndarray | None = None,
    joint_indices: np.ndarray | None = None,
    rest_qpos: np.ndarray | None = None,
    max_iters: int = 100,
    pos_tol: float = 2e-4,
    rot_tol: float = 2e-3,
    damping: float = 5e-2,
    min_damping: float = 2e-3,
    max_step: float = 0.20,
    nullspace_gain: float = 0.05,
    rot_weight: float = 1.0,
) -> IKResult:
    """Solve for arm joint angles placing ``site_id`` at the target pose.

    Operates **in place** on ``data`` — pass a scratch ``MjData`` if you do not
    want the simulation state disturbed. ``PandaRobot.solve_ik`` does this for you.

    Parameters
    ----------
    dof_indices, qpos_indices
        Which velocity-space and position-space entries the solver may move.
        For the Panda these coincide (7 hinge joints) but keeping them separate
        means the same routine works on models with free or ball joints.
    damping, min_damping
        Levenberg-Marquardt style adaptive damping. Damping is scaled down as the
        residual shrinks: heavy damping far from the target keeps the step stable
        through singularities, but that same damping biases the fixed point and
        leaves a sub-millimetre residual the solver can never remove. Relaxing it
        near convergence recovers the accuracy without giving up the stability.
    rest_qpos
        Posture the null space is biased toward. Without this the redundant DoF
        drifts, and over a long trajectory the elbow slowly winds into a joint
        limit — a failure that looks like "IK randomly stopped working".
    """
    if dof_indices is None:
        dof_indices = np.arange(7)
    if qpos_indices is None:
        qpos_indices = np.arange(7)
    if joint_indices is None:
        joint_indices = np.arange(7)

    n = len(dof_indices)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = np.zeros(6)

    converged = False
    pos_err = rot_err = np.inf
    it = 0
    for it in range(1, max_iters + 1):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        err[:3] = target_pos - data.site_xpos[site_id]
        if target_quat is not None:
            err[3:] = quat_error(target_quat, data.site_xmat[site_id].reshape(3, 3)) * rot_weight
        else:
            err[3:] = 0.0

        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]) / max(rot_weight, 1e-9))
        if pos_err < pos_tol and (target_quat is None or rot_err < rot_tol):
            converged = True
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, dof_indices], jacr[:, dof_indices]])
        if target_quat is None:
            jac = jac[:3]
            e = err[:3]
        else:
            e = err

        # dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e, with λ annealed by the current residual.
        residual = float(np.linalg.norm(e))
        lam = max(min_damping, min(damping, damping * residual / 0.05))
        jjt = jac @ jac.T
        jjt[np.diag_indices_from(jjt)] += lam**2
        dq = jac.T @ np.linalg.solve(jjt, e)

        if rest_qpos is not None and nullspace_gain > 0.0:
            # Project a pull toward the rest posture through the null space of J,
            # so it never fights the Cartesian objective.
            jac_pinv = jac.T @ np.linalg.solve(jjt, np.eye(jac.shape[0]))
            null_proj = np.eye(n) - jac_pinv @ jac
            # Fade the posture bias out as we converge; a constant pull competes
            # with the last few micrometres of Cartesian error.
            gain = nullspace_gain * min(1.0, residual / 0.02)
            dq += null_proj @ (gain * (rest_qpos - data.qpos[qpos_indices]))

        norm = np.linalg.norm(dq)
        if norm > max_step:
            dq *= max_step / norm

        data.qpos[qpos_indices] += dq
        # Respect joint limits every iteration rather than only at the end:
        # clamping once at the end can leave a pose that never satisfied the target.
        lo = model.jnt_range[joint_indices, 0]
        hi = model.jnt_range[joint_indices, 1]
        limited = model.jnt_limited[joint_indices].astype(bool)
        data.qpos[qpos_indices] = np.where(
            limited, np.clip(data.qpos[qpos_indices], lo, hi), data.qpos[qpos_indices]
        )

    return IKResult(
        qpos=data.qpos[qpos_indices].copy(),
        pos_err=pos_err,
        rot_err=rot_err,
        iterations=it,
        converged=converged,
    )
