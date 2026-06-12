CONTEXT SPECIFICATION: PHASE IV NONLINEAR OPTIMAL CONTROL

1. Core Architecture & VariablesThe coding agent must strictly preserve the 7-dimensional state vector, 7-dimensional costate vector, and 2-dimensional control vector using 0-based indexing.

1.1 State Vector ($x \in \mathbb{R}^7$)x[0] : $p_x$ (Horizontal position, m)x[1] : $p_z$ (Vertical position/Altitude, m)x[2] : $v_x$ (Horizontal velocity, m/s)x[3] : $v_z$ (Vertical velocity, m/s)x[4] : $\theta$ (Pitch angle, rad)x[5] : $\omega$ (Angular velocity, rad/s)x[6] : $m$ (Current mass, kg)

1.2 Costate Vector ($p \in \mathbb{R}^7$)p[0] to p[6] correspond directly as the adjoint variables for states x[0] to x[6].

2. The Nonlinear Dynamics Engine ($f_{nl}$)Unlike previous phases, the agent must not use the $A(t)$ or $B(t)$ Jacobians for the plant dynamics. The true nonlinear equations of motion must be implemented:Pythondef nonlinear_dynamics(t, x, u, params):
    # x: state vector, u: control vector [T, tau]
    px, pz, vx, vz, theta, omega, m = x
    T, tau = u
    g, I, alpha = params['g'], params['I'], params['alpha']
    
    x_dot = np.zeros_like(x)
    x_dot[0] = vx
    x_dot[1] = vz
    x_dot[2] = -(T / m) * np.sin(theta)
    x_dot[3] = (T / m) * np.cos(theta) - g
    x_dot[4] = omega
    x_dot[5] = tau / I
    x_dot[6] = -alpha * T
    return x_dot


3. Scenario A: Linear LQR on Nonlinear PlantObjective: Test the robustness of the Phase I linear controller when subjected to true nonlinear physics at varying initial angle deviations.

3.1 Implementation StepsLoad LQR Gains: Load the pre-calculated time-varying feedback gain matrix interpolator $K(t)$ from Phase I.Forward Integration: Use scipy.integrate.solve_ivp.Control Law: Inside the ODE derivative function, calculate the unconstrained control $u_{req} = -K(t) \cdot x(t)$. Saturation: Apply np.clip to enforce physical actuator limits ($[T_{min}, T_{max}]$ and $[\tau_{min}, \tau_{max}]$).Feed to Plant: Pass the clipped $u_{act}$ into the nonlinear_dynamics function.


4. Scenario B: Full Nonlinear TPBVP (PMP)Objective: Solve the exact nonlinear Two-Point Boundary Value Problem to find the globally optimal trajectory without small-angle assumptions.

4.1 The Nonlinear Costate EquationsThe agent must implement the exact analytical partial derivatives $-\frac{\partial H}{\partial x}$. The costate ODE vector ($\dot{p}$) is:Pythondef nonlinear_costate_dynamics(t, x, p, u, Q, params):
    px, pz, vx, vz, theta, omega, m = x
    T, tau = u
    
    p_dot = np.zeros_like(p)
    p_dot[0] = -2 * Q[0,0] * px
    p_dot[1] = -2 * Q[1,1] * pz
    p_dot[2] = -2 * Q[2,2] * vx - p[0]
    p_dot[3] = -2 * Q[3,3] * vz - p[1]
    p_dot[4] = -2 * Q[4,4] * theta + (T / m) * (p[2] * np.cos(theta) + p[3] * np.sin(theta))
    p_dot[5] = -2 * Q[5,5] * omega - p[4]
    p_dot[6] = -2 * Q[6,6] * m     + (T / m**2) * (p[3] * np.cos(theta) - p[2] * np.sin(theta))
    return p_dot

4.2 The Nonlinear Stationarity Condition (Optimal Control Law)The agent must define a function to compute $u^*(t)$ continuously based on the current $x(t)$ and $p(t)$, derived from $\frac{\partial H}{\partial u} = 0$:Pythondef compute_optimal_control(x, p, R, params):
    theta, m = x[4], x[6]
    alpha = params['alpha']
    
    # 1. Calculate Unconstrained Controls
    T_req = (1 / (2 * R[0,0])) * (p[2] * (np.sin(theta) / m) - p[3] * (np.cos(theta) / m) + p[6] * alpha)
    tau_req = -(p[5]) / (2 * params['I'] * R[1,1])
    
    # 2. Enforce Physical Actuator Limits
    T_opt = np.clip(T_req, params['T_min'], params['T_max'])
    tau_opt = np.clip(tau_req, params['tau_min'], params['tau_max'])
    
    return np.array([T_opt, tau_opt])



5. BVP Solver Configuration (scipy.integrate.solve_bvp)

5.1 The Combined ODE SystemThe agent must wrap states and costates into a 14-dimensional vector y.Pythondef bvp_system(t, y, params, Q, R):
    # y is shape (14, N)
    x = y[0:7]
    p = y[7:14]
    
    # Calculate optimal control across all time steps
    u = compute_optimal_control(x, p, R, params)
    
    # Compute derivatives
    dx_dt = nonlinear_dynamics(t, x, u, params)
    dp_dt = nonlinear_costate_dynamics(t, x, p, u, Q, params)
    
    return np.vstack((dx_dt, dp_dt))

5.2 Boundary Conditions (bc function)Define the initial states and the terminal transversality conditions:Pythondef bvp_boundaries(ya, yb, x0, Qf):
    # ya: initial states/costates (at t=0)
    # yb: terminal states/costates (at t=tf)
    
    # Initial state must match x0
    res_initial = ya[0:7] - x0
    
    # Transversality: p(tf) = d(Phi)/dx = 2 * Qf * x(tf)
    p_tf_expected = 2 * Qf @ yb[0:7]
    res_terminal = yb[7:14] - p_tf_expected
    
    return np.concatenate((res_initial, res_terminal))

5.3 The Initial Guess (CRITICAL SUCCESS FACTOR)Nonlinear TPBVPs will fail to converge with a generic zero-guess. The agent must use the resulting state trajectory $x_{LQR}(t)$ from Scenario A as the initial guess for the BVP solver. For the costate initial guess, compute $p_{guess}(t) = P(t) x_{LQR}(t)$, where $P(t)$ is the Riccati matrix from Phase I.6. Validation and Output RequirementsThe agent must script the following explicit validation outputs to satisfy rubric requirements:Comparative Plots: A side-by-side (or overlaid) plot of $p_x$ vs $p_z$ for Scenario A vs. Scenario B. Attitude Vectoring (Quiver Plot): The $p_x$ vs $p_z$ plot MUST include a matplotlib.pyplot.quiver overlay demonstrating the pitch angle ($\theta$) at periodic time steps. Shooting Residuals Report: After solve_bvp completes, print the absolute maximum value of the evaluated bc function to prove boundary convergence to $< 10^{-6}$. Energy Comparison: Compute and print the total control effort $J_u = \int (u^\top R u) dt$ for both scenarios to prove that the nonlinear controller discovered a more efficient path for large-deviation cases.