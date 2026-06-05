import numpy as np
import control as ct
from scipy.integrate import solve_ivp

def build_jacobians(m, params):
    """
    Core function to populate the A and B matrices given a specific mass.
    The underscore denotes this is an internal helper function.
    """
    g = params['g']
    α = params['α']
    I = params['I']
    
    A = np.zeros((7, 7))
    B = np.zeros((7, 2))
    
    # --- A Matrix ---
    A[0, 2] = 1.0        # p_x_dot wrt v_x
    A[1, 3] = 1.0        # p_z_dot wrt v_z
    A[4, 5] = 1.0        # theta_dot wrt omega
    A[2, 4] = g          # v_x_dot wrt theta
    A[3, 6] = -g / m     # v_z_dot wrt m (Uses the dynamically passed 'm')
    
    # --- B Matrix ---
    B[3, 0] = 1.0 / m    # v_z_dot wrt delta_T (Uses the dynamically passed 'm')
    B[5, 1] = 1.0 / I    # omega_dot wrt tau
    B[6, 0] = -α     # m_dot wrt delta_T
    
    return A, B


def get_hover_dynamics(params):
    """
    Returns the LTI A and B matrices for the hover equilibrium.
    """
    return build_jacobians(params['m0'], params) # Hover mass is just the initial constant mass

def get_descent_dynamics(t, params):
    """LTV dynamics. Uses 't' to calculate mass depletion."""
    m_t = params['m0'] * np.exp(-params['alpha'] * params['g'] * t)
    return build_jacobians(m_t, params)


def gramian_ode(t, W_flat, dynamics_func, params):
    """
    The matrix ODE for the Controllability Gramian, flattened for solve_ivp.
    dW/dt = A(t)W + WA(t)^T + B(t)B(t)^T
    """
    W = W_flat.reshape((7, 7))  # 1. Reshape the 1D vector back into a 7x7 matrix
    A, B = dynamics_func(t, params)  # 2. Get the time-varying matrices for this specific time step
    dW_dt = A @ W + W @ A.T + B @ B.T  # 3. Compute the matrix derivative
    return dW_dt.flatten()


def analyze_controllability(dynamics_func, t_eval, t_f, params, mode_name="System"):
    """
    Computes SVD for both the algebraic controllability matrix at t_eval
    and the finite-horizon Gramian over [0, t_f].
    """
    print(f"\n--- Controllability Analysis: {mode_name} ---")
    
    # 1. Algebraic Controllability (Snapshot at t_eval)
    A_snap, B_snap = dynamics_func(t_eval, params)
    C_mat = ct.ctrb(A_snap, B_snap)
    svd_C = np.linalg.svd(C_mat, compute_uv=False)
    
    print(f"Algebraic Matrix Rank at t={t_eval}s: {np.linalg.matrix_rank(C_mat)} / 7")
    print(f"Algebraic Singular Values:\n{np.array2string(svd_C, precision=4, suppress_small=True)}")
    
    # 2. Finite-Horizon Controllability Gramian (Integration over [0, t_f])
    sol = solve_ivp(
        _gramian_ode, 
        [0, t_f], 
        np.zeros(49), 
        args=(dynamics_func, params),
        method='RK45'
    )
    W_gramian = sol.y[:, -1].reshape((7, 7))
    svd_W = np.linalg.svd(W_gramian, compute_uv=False)
    
    print(f"Gramian Matrix Rank over {t_f}s: {np.linalg.matrix_rank(W_gramian)} / 7")
    print(f"Gramian Singular Values:\n{np.array2string(svd_W, precision=4, suppress_small=True)}")
    
    return svd_C, svd_W