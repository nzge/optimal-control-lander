
g = 9.8 #m/s
I = 10 #kg-m/s^2
m0 = 20 #kg
m_dry = 8 #kg, propellant depleted mass floor
T_min = 0 #N
T_max = 300 #N
α = 0.005 #kg/(N-s)
τ_min = -30 #N-m
τ_max = 30 #N-m

# Part I.5.2 descent trim (weak horizontal coupling regime)
tf_descent = 20.0  # s, mission end time
vx_descent = 40.0  # m/s, large horizontal speed
vz_descent = -8.0  # m/s, descending
theta_descent = 0.0  # rad, nearly vertical

# Part III landing platform (manifold M2, hover-trim deviation coords)
p_c = 12.0  # m — scaled for LTI-valid mission arc (spec uses 100 m at larger scale)
r_platform = 5.0  # m
tf_mission_max = 60.0  # s, upper bound on total mission time

params = {
    "g": g,
    "I": I,
    "m0": m0,
    "m_dry": m_dry,
    "T_min": T_min,
    "T_max": T_max,
    "α": α,
    "τ_min": τ_min,
    "τ_max": τ_max,
    "tf_descent": tf_descent,
    "vx_descent": vx_descent,
    "vz_descent": vz_descent,
    "theta_descent": theta_descent,
    "p_c": p_c,
    "r_platform": r_platform,
    "tf_mission_max": tf_mission_max,
}