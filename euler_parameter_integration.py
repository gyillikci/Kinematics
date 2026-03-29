import numpy as np
from scipy.integrate import solve_ivp

def euler_parameter_ode(t, beta):
    """
    Euler Parameter differential kinematic equation from slide 71:
    β_dot = (1/2) * [B(β)] * [0, ω1, ω2, ω3]^T
    """
    beta0, beta1, beta2, beta3 = beta

    # Angular velocity in deg/s, convert to rad/s
    omega_deg = 20.0  # magnitude in deg/s
    omega_rad = np.deg2rad(omega_deg)

    # Body angular velocity components
    omega1 = np.sin(0.1 * t) * omega_rad
    omega2 = 0.01 * omega_rad
    omega3 = np.cos(0.1 * t) * omega_rad

    # Matrix from slide 71
    B_matrix = np.array([
        [beta0, -beta1, -beta2, -beta3],
        [beta1,  beta0, -beta3,  beta2],
        [beta2,  beta3,  beta0, -beta1],
        [beta3, -beta2,  beta1,  beta0]
    ])

    # Omega vector (with 0 as first element)
    omega_vec = np.array([0, omega1, omega2, omega3])

    # Compute derivative
    beta_dot = 0.5 * B_matrix @ omega_vec

    return beta_dot

# Initial conditions
beta_0 = np.array([0.408248, 0.0, 0.408248, 0.816497])

# Time span
t_span = (0, 42)
t_eval = np.linspace(0, 42, 1000)

# Integrate using RK45
solution = solve_ivp(euler_parameter_ode, t_span, beta_0, method='RK45',
                     t_eval=t_eval, rtol=1e-10, atol=1e-12)

# Get final values at t = 42 seconds
beta_final = solution.y[:, -1]

# Normalize to maintain unit quaternion (optional, for numerical stability check)
beta_norm_full = np.sqrt(np.sum(beta_final**2))

print("=" * 60)
print("Euler Parameter Integration Results")
print("=" * 60)
print(f"\nInitial β(0) = {beta_0}")
print(f"Initial norm = {np.sqrt(np.sum(beta_0**2)):.6f}")
print(f"\nFinal β(42) = {beta_final}")
print(f"Full quaternion norm |β| = {beta_norm_full:.6f}")

# Compute the requested norm: sqrt(β₁² + β₂² + β₃²)
beta1, beta2, beta3 = beta_final[1], beta_final[2], beta_final[3]
vector_norm = np.sqrt(beta1**2 + beta2**2 + beta3**2)

print(f"\n" + "=" * 60)
print(f"ANSWER: sqrt(β₁² + β₂² + β₃²) at t=42s = {vector_norm:.6f}")
print("=" * 60)
