"""
Control saturation and numerically enforced state constraints for LQR rollouts.

Control bounds (PMP): u* = argmin_{u in U} H  =>  saturate unconstrained law.
State bounds (numerical): truncate forward simulation at first violation of
  p_z >= 0  and  m >= m_dry  in absolute coordinates reconstructed from trim.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from dynamics import descent_trim, hover_trim

IDX_PZ = 1
IDX_M = 6


@dataclass
class RolloutConstraints:
    """Configuration for constrained closed-loop simulation."""

    params: dict
    trim_func: Callable = descent_trim
    enforce_control: bool = True
    enforce_state: bool = True


@dataclass
class RolloutConstraintInfo:
    """Diagnostics from a constrained rollout."""

    control_saturated: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    u_req_hist: Optional[np.ndarray] = None
    saturation_fraction: float = 0.0
    state_violated: bool = False
    violation_time: Optional[float] = None
    violation_name: Optional[str] = None
    terminated_early: bool = False


def trim_for_dynamics(dynamics_func):
    """Return the nominal trim callback matching a dynamics linearization."""
    if dynamics_func.__name__ == "get_hover_dynamics":
        return hover_trim
    return descent_trim


def control_bounds(t, params, trim_func=descent_trim):
    """
    Box bounds on u = [delta_T, tau] at time t.

    delta_T is thrust deviation from trim; tau is absolute torque.
    """
    trim = trim_func(t, params)
    delta_T_lo = params["T_min"] - trim["T"]
    delta_T_hi = params["T_max"] - trim["T"]
    return np.array([delta_T_lo, params["τ_min"]]), np.array([delta_T_hi, params["τ_max"]])


def saturate_control(u_req, t, params, trim_func=descent_trim):
    """PMP saturation: clip unconstrained control to the physical control set."""
    lo, hi = control_bounds(t, params, trim_func)
    return np.clip(u_req, lo, hi)


def absolute_state(x_dev, t, params, trim_func=descent_trim):
    """Reconstruct absolute state from trim + deviation coordinates."""
    trim = trim_func(t, params)
    return np.array(
        [
            trim["px"] + x_dev[0],
            trim["pz"] + x_dev[1],
            trim["vx"] + x_dev[2],
            trim["vz"] + x_dev[3],
            trim["theta"] + x_dev[4],
            trim["omega"] + x_dev[5],
            trim["m"] + x_dev[6],
        ]
    )


def state_margins(x_dev, t, params, trim_func=descent_trim):
    """
    Constraint margins in absolute coordinates (positive => feasible).

    Returns dict mapping constraint name to margin.
    """
    x_abs = absolute_state(x_dev, t, params, trim_func)
    m_dry = params.get("m_dry", 0.0)
    return {
        "p_z": x_abs[IDX_PZ],
        "m": x_abs[IDX_M] - m_dry,
    }


def first_state_violation(x_dev, t, params, trim_func=descent_trim):
    """Return (name, margin) of first violated constraint, or (None, None)."""
    for name, margin in state_margins(x_dev, t, params, trim_func).items():
        if margin < 0.0:
            return name, margin
    return None, None


def apply_control(u_req, t, constraints: RolloutConstraints):
    """Apply control saturation if enabled."""
    if not constraints.enforce_control:
        return u_req.copy(), False
    u_act = saturate_control(u_req, t, constraints.params, constraints.trim_func)
    return u_act, not np.allclose(u_req, u_act)


def check_state(x, t, constraints: RolloutConstraints):
    """Return violation name if state constraints are enabled and violated."""
    if not constraints.enforce_state:
        return None
    name, _ = first_state_violation(x, t, constraints.params, constraints.trim_func)
    return name


def empty_info(n_steps, n_ctrl):
    return RolloutConstraintInfo(
        control_saturated=np.zeros(n_steps, dtype=bool),
        u_req_hist=np.zeros((n_steps, n_ctrl)) if n_ctrl else None,
    )


def finalize_info(info: RolloutConstraintInfo):
    if info.control_saturated.size:
        info.saturation_fraction = float(np.mean(info.control_saturated))
    return info
