import numpy as np
import control as ct
from scipy.integrate import solve_ivp

# State order: [p_x, p_z, v_x, v_z, theta, omega, m]


def descent_trim(t, params):
    """
    Nominal powered-descent trim from Part I.5.2:
    theta* ~ 0, v_x* >> 0, v_z* < 0, T* ~ m(t) g.
    """
    g = params["g"]
    alpha = params["α"]
    m0 = params["m0"]
    tf = params["tf_descent"]

    m = m0 * np.exp(-alpha * g * t)
    theta = params.get("theta_descent", 0.0)
    T = m * g / np.cos(theta)

    # Optional slow bleed-off of horizontal speed over the mission window
    frac = t / tf if tf > 0 else 0.0
    vx = params["vx_descent"] * (1.0 - 0.5 * frac)
    vz = params["vz_descent"]

    return {
        "px": 0.0,
        "pz": 0.0,
        "vx": vx,
        "vz": vz,
        "theta": theta,
        "omega": 0.0,
        "m": m,
        "T": T,
        "tau": 0.0,
    }


def linearize_at_trim(trim, params):
    """
    Linearize x_dot = f(x, u) about trim with control u = [delta_T, tau],
    delta_T = T - T*.
    """
    g = params["g"]
    alpha = params["α"]
    I = params["I"]

    m = trim["m"]
    theta = trim["theta"]
    T = trim["T"]

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    A = np.zeros((7, 7))
    B = np.zeros((7, 2))

    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[4, 5] = 1.0

    # v_x = (T/m) sin(theta)
    A[2, 4] = T * cos_t / m
    A[2, 6] = -T * sin_t / (m * m)
    B[2, 0] = sin_t / m

    # v_z = (T/m) cos(theta) - g
    A[3, 4] = -T * sin_t / m
    A[3, 6] = -T * cos_t / (m * m)
    B[3, 0] = cos_t / m

    B[5, 1] = 1.0 / I
    B[6, 0] = -alpha

    return A, B


def build_jacobians(m, params, theta=0.0):
    """Backward-compatible helper: hover-style trim with mass m."""
    trim = {
        "px": 0.0,
        "pz": 0.0,
        "vx": 0.0,
        "vz": 0.0,
        "theta": theta,
        "omega": 0.0,
        "m": m,
        "T": m * params["g"] / np.cos(theta),
        "tau": 0.0,
    }
    return linearize_at_trim(trim, params)


def get_hover_dynamics(t, params):
    """LTI hover linearization at (v=0, theta=0, T=mg, m=m0)."""
    return build_jacobians(params["m0"], params, theta=0.0)


def get_descent_dynamics(t, params):
    """LTV linearization along the nominal descent trim at time t."""
    return linearize_at_trim(descent_trim(t, params), params)


def gramian_ode(t, W_flat, dynamics_func, params):
    """dW/dt = A(t)W + W A(t)^T + B(t)B(t)^T"""
    W = W_flat.reshape((7, 7))
    A, B = dynamics_func(t, params)
    dW_dt = A @ W + W @ A.T + B @ B.T
    return dW_dt.flatten()

def finite_horizon_gramian(dynamics_func, t0, t1, params):
    sol = solve_ivp(
        gramian_ode,
        [t0, t1],
        np.zeros(49),
        args=(dynamics_func, params),
        method="RK45",
    )
    return sol.y[:, -1].reshape((7, 7))

    
def controllability_matrix(A, B):
    return ct.ctrb(A, B)


def singular_values(matrix):
    return np.linalg.svd(matrix, compute_uv=False)


def sigma_min_controllable(singulars, tol=1e-8):
    """Smallest singular value above numerical zero (excludes mass mode)."""
    controllable = singulars[singulars > tol]
    return controllable.min() if controllable.size else 0.0


def sigma_structural_min(singulars):
    """
    Second-smallest singular value (6th of 7).

    The smallest (7th) mode is the near-zero mass direction: m_dot = -alpha * delta_T
    only allows fuel burn, not replenishment, so that mode is structurally
  uncontrollable. The 6th mode is the weakest *physically meaningful* direction.
    """
    return singulars[-2] if singulars.size >= 2 else 0.0


def horizontal_energy_index(A, B, horizon, dt=0.02):
    """
    Proxy for horizontal controllability: energy to reach unit p_x perturbation
  over a fixed horizon using the chain torque -> theta -> v_x -> p_x.
    """
    n = A.shape[0]
    Phi = np.eye(n)
    gram = np.zeros((n, n))
    t = 0.0
    while t < horizon:
        Phi_dt = Phi.copy()
        k1 = A @ Phi_dt
        k2 = A @ (Phi_dt + 0.5 * dt * k1)
        k3 = A @ (Phi_dt + 0.5 * dt * k2)
        k4 = A @ (Phi_dt + dt * k3)
        Phi = Phi_dt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        gram += Phi @ B @ B.T @ Phi.T * dt
        t += dt
    e_px = np.zeros(n)
    e_px[0] = 1.0
    denom = e_px @ gram @ e_px
    return 1.0 / denom if denom > 1e-16 else np.inf


def analyze_controllability(dynamics_func, t_eval, t_f, params, mode_name="System"):
    """Snapshot controllability matrix and Gramian over [0, t_f]."""
    print(f"\n--- Controllability Analysis: {mode_name} ---")

    A_snap, B_snap = dynamics_func(t_eval, params)
    C_mat = controllability_matrix(A_snap, B_snap)
    svd_C = singular_values(C_mat)

    print(f"Algebraic Matrix Rank at t={t_eval}s: {np.linalg.matrix_rank(C_mat)} / 7")
    print(f"Algebraic Singular Values:\n{np.array2string(svd_C, precision=4, suppress_small=True)}")

    W_gramian = finite_horizon_gramian(dynamics_func, 0.0, t_f, params)
    svd_W = singular_values(W_gramian)

    print(f"Gramian Matrix Rank over {t_f}s: {np.linalg.matrix_rank(W_gramian)} / 7")
    print(f"Gramian Singular Values:\n{np.array2string(svd_W, precision=4, suppress_small=True)}")

    return svd_C, svd_W


def compare_along_trajectory(params, times=None):
    """
    Compare hover vs descent controllability at multiple trim times.
    Uses remaining-horizon Gramian [t, tf] — the metric that reveals deterioration.
    """
    tf = params["tf_descent"]
    if times is None:
        times = np.linspace(0.0, tf, 9, endpoint=False)

    rows = []
    for t in times:
        t1 = tf
        remaining = max(t1 - t, 1e-3)

        A_h, B_h = get_hover_dynamics(t, params)
        A_d, B_d = get_descent_dynamics(t, params)

        sC_h = singular_values(controllability_matrix(A_h, B_h))
        sC_d = singular_values(controllability_matrix(A_d, B_d))

        W_h = finite_horizon_gramian(get_hover_dynamics, t, t1, params)
        W_d = finite_horizon_gramian(get_descent_dynamics, t, t1, params)
        sW_h = singular_values(W_h)
        sW_d = singular_values(W_d)

        rows.append(
            {
                "t": t,
                "remaining": remaining,
                "sigma_min_C_hover": sigma_min_controllable(sC_h),
                "sigma_min_C_descent": sigma_min_controllable(sC_d),
                "sigma_struct_W_hover": sigma_structural_min(sW_h),
                "sigma_struct_W_descent": sigma_structural_min(sW_d),
                "horiz_index_descent": horizontal_energy_index(A_d, B_d, remaining),
                "trim": descent_trim(t, params),
            }
        )

    return rows
