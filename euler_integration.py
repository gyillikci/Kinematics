#!/usr/bin/env python3
"""
Numerical Integration of (3-2-1) Euler Angle Kinematic Equations

Given:
- Initial (3-2-1) Euler angles: (ψ, θ, φ) = (40, 30, 80) degrees
- Body angular velocity: ω = [sin(0.1t), 0.01, cos(0.1t)]ᵀ × 20 deg/s
- Simulation time: 60 seconds
- Find: Euler angle norm at t = 42s
"""

import numpy as np
from scipy.integrate import odeint
import math


def euler_321_kinematics(angles, t):
    """
    Differential kinematic equations for (3-2-1) Euler angles.

    angles = [psi, theta, phi] in radians

    The kinematic differential equation is:
    [ψ̇]       1    [0      sinφ        cosφ    ] [ω₁]
    [θ̇] = ------- [0    cosφ·cosθ   -sinφ·cosθ] [ω₂]
    [φ̇]    cosθ   [cosθ  sinφ·sinθ   cosφ·sinθ] [ω₃]

    Or equivalently:
    ψ̇ = (sinφ·ω₂ + cosφ·ω₃) / cosθ
    θ̇ = cosφ·ω₂ - sinφ·ω₃
    φ̇ = ω₁ + tanθ·(sinφ·ω₂ + cosφ·ω₃)
    """
    psi, theta, phi = angles

    # Angular velocity in body frame (deg/s), convert to rad/s
    omega_deg = np.array([
        np.sin(0.1 * t),
        0.01,
        np.cos(0.1 * t)
    ]) * 20.0  # deg/s

    omega = np.radians(omega_deg)  # rad/s

    omega1, omega2, omega3 = omega

    # Euler angle rates
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Kinematic equations for (3-2-1)
    psi_dot = (sin_phi * omega2 + cos_phi * omega3) / cos_theta
    theta_dot = cos_phi * omega2 - sin_phi * omega3
    phi_dot = omega1 + (sin_phi * omega2 + cos_phi * omega3) * sin_theta / cos_theta

    return [psi_dot, theta_dot, phi_dot]


def main():
    # Initial conditions in degrees
    psi_0 = 40.0    # yaw
    theta_0 = 30.0  # pitch
    phi_0 = 80.0    # roll

    # Convert to radians
    angles_0 = np.radians([psi_0, theta_0, phi_0])

    print("=" * 60)
    print("Euler Angle Numerical Integration")
    print("=" * 60)

    print("\nInitial (3-2-1) Euler angles:")
    print(f"  ψ (yaw)   = {psi_0}° = {angles_0[0]:.6f} rad")
    print(f"  θ (pitch) = {theta_0}° = {angles_0[1]:.6f} rad")
    print(f"  φ (roll)  = {phi_0}° = {angles_0[2]:.6f} rad")

    print("\nAngular velocity (body frame):")
    print("  ω = [sin(0.1t), 0.01, cos(0.1t)]ᵀ × 20 deg/s")

    # Time array - simulate for 60 seconds with small time step
    dt = 0.01  # 10 ms time step
    t = np.arange(0, 60 + dt, dt)

    # Integrate
    print("\nIntegrating over 60 seconds...")
    solution = odeint(euler_321_kinematics, angles_0, t)

    # Find index closest to t = 42s
    idx_42 = np.argmin(np.abs(t - 42.0))
    t_42 = t[idx_42]

    # Get angles at t = 42s (in radians, not wrapped)
    psi_42, theta_42, phi_42 = solution[idx_42]

    print(f"\nAt t = {t_42}s:")
    print(f"  ψ = {psi_42:.6f} rad ({np.degrees(psi_42):.4f}°)")
    print(f"  θ = {theta_42:.6f} rad ({np.degrees(theta_42):.4f}°)")
    print(f"  φ = {phi_42:.6f} rad ({np.degrees(phi_42):.4f}°)")

    # Compute Euler angle norm
    euler_norm = np.sqrt(psi_42**2 + theta_42**2 + phi_42**2)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"\nEuler angle norm at t = 42s:")
    print(f"  √(ψ² + θ² + φ²) = √({psi_42:.6f}² + {theta_42:.6f}² + {phi_42:.6f}²)")
    print(f"                  = {euler_norm:.6f} rad")

    # Also show some intermediate values
    print("\n" + "-" * 60)
    print("Time history at selected points:")
    print("-" * 60)
    print(f"{'Time (s)':<10} {'ψ (rad)':<12} {'θ (rad)':<12} {'φ (rad)':<12} {'Norm (rad)':<12}")
    print("-" * 60)

    for t_check in [0, 10, 20, 30, 40, 42, 50, 60]:
        idx = np.argmin(np.abs(t - t_check))
        psi_t, theta_t, phi_t = solution[idx]
        norm_t = np.sqrt(psi_t**2 + theta_t**2 + phi_t**2)
        print(f"{t[idx]:<10.1f} {psi_t:<12.6f} {theta_t:<12.6f} {phi_t:<12.6f} {norm_t:<12.6f}")


if __name__ == "__main__":
    main()
