#!/usr/bin/env python3
"""
Visualization for Euler Angle Problems

Problem 1: Elaboration on (2-3-2) Euler angle differential kinematics
Problem 2: Animation of Euler angle numerical integration
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io
import math


# =============================================================================
# Rotation Matrix Functions
# =============================================================================

def Rx(angle):
    """Rotation about X-axis (1-axis)"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def Ry(angle):
    """Rotation about Y-axis (2-axis)"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def Rz(angle):
    """Rotation about Z-axis (3-axis)"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

def Euler3212C(psi, theta, phi):
    """Convert (3-2-1) Euler angles to DCM"""
    return Rx(phi) @ Ry(theta) @ Rz(psi)

def Euler2322C(theta1, theta2, theta3):
    """Convert (2-3-2) Euler angles to DCM"""
    return Ry(theta3) @ Rz(theta2) @ Ry(theta1)


# =============================================================================
# Drawing Functions
# =============================================================================

def draw_frame(ax, dcm, label, colors=['red', 'green', 'blue'], length=1.0, lw=2.5, alpha=1.0):
    """Draw coordinate frame axes"""
    for i, (c, axis_name) in enumerate(zip(colors, ['x', 'y', 'z'])):
        axis = dcm.T[:, i] * length
        ax.quiver(0, 0, 0, axis[0], axis[1], axis[2],
                  color=c, linewidth=lw, arrow_length_ratio=0.12, alpha=alpha)
        tip = axis * 1.15
        ax.text(tip[0], tip[1], tip[2], f'{label}_{axis_name}',
                color=c, fontsize=8, fontweight='bold', alpha=alpha)

def draw_angular_velocity(ax, omega, dcm, scale=0.3, color='purple'):
    """Draw angular velocity vector in body frame"""
    # Transform omega to inertial frame for visualization
    omega_inertial = dcm.T @ omega
    omega_normalized = omega_inertial / (np.linalg.norm(omega_inertial) + 1e-10) * scale
    ax.quiver(0, 0, 0, omega_normalized[0], omega_normalized[1], omega_normalized[2],
              color=color, linewidth=3, arrow_length_ratio=0.2, alpha=0.8)

def setup_ax(ax, limit=1.4):
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)

def save_frame(fig, frames):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    buf.close()


# =============================================================================
# Problem 1: (2-3-2) Euler Angle Differential Kinematics Elaboration
# =============================================================================

def elaborate_problem1():
    """
    Create visualization explaining (2-3-2) Euler angle differential kinematics.
    """
    print("Creating Problem 1 elaboration...")

    fig = plt.figure(figsize=(16, 12))

    # Main title
    fig.suptitle('Problem 1: (2-3-2) Euler Angle Differential Kinematic Equation',
                 fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: Rotation sequence diagram
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.set_title('(2-3-2) Rotation Sequence\nθ₁=30°, θ₂=45°, θ₃=60°', fontsize=10, fontweight='bold')

    theta1, theta2, theta3 = np.radians([30, 45, 60])

    # Show intermediate frames
    C0 = np.eye(3)  # N frame
    C1 = Ry(theta1)  # After first Y rotation
    C2 = Rz(theta2) @ C1  # After Z rotation
    C3 = Ry(theta3) @ C2  # Final B frame

    draw_frame(ax1, C0, 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.6, 1.5, 0.3)
    draw_frame(ax1, C3, 'B', ['red', 'green', 'blue'], 1.0, 2.5, 1.0)
    setup_ax(ax1)
    ax1.view_init(elev=20, azim=45)

    # Panel 2: Explanation text
    ax2 = fig.add_subplot(232)
    ax2.axis('off')
    ax2.set_title('Understanding the Equation', fontsize=11, fontweight='bold')

    explanation = """
    For (2-3-2) Euler angles (θ₁, θ₂, θ₃):

    Rotation Sequence:
    1. Rotate θ₁ about Y-axis (2-axis)
    2. Rotate θ₂ about Z'-axis (3-axis)
    3. Rotate θ₃ about Y''-axis (2-axis)

    The kinematic equation relates angular
    velocity ω to Euler angle rates θ̇:

         ω = [B(θ)] · θ̇

    Therefore:
         θ̇ = [B(θ)]⁻¹ · ω

    Key Properties:
    • Symmetric sequence (2-3-2)
    • Singularity at θ₂ = 0° or 180°
      (when sin(θ₂) = 0)
    • Factor 1/sin(θ₂) in the equation
    """
    ax2.text(0.05, 0.95, explanation, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 3: The B-inverse matrix
    ax3 = fig.add_subplot(233)
    ax3.axis('off')
    ax3.set_title('The [B]⁻¹ Matrix for (2-3-2)', fontsize=11, fontweight='bold')

    matrix_text = """
    ┌     ┐       1    ┌                                    ┐ ┌    ┐
    │ θ̇₁  │    ────── │  cθ₃      0        sθ₃              │ │ ω₁ │
    │     │           │                                     │ │    │
    │ θ̇₂  │ =   1     │ -sθ₃·sθ₂   0       cθ₃·sθ₂          │ │ ω₂ │
    │     │   ─────   │                                     │ │    │
    │ θ̇₃  │   sθ₂     │ -cθ₃·cθ₂  sθ₂     -sθ₃·cθ₂          │ │ ω₃ │
    └     ┘           └                                    ┘ └    ┘

    Where:
    • cθ₂ = cos(θ₂),  sθ₂ = sin(θ₂)
    • cθ₃ = cos(θ₃),  sθ₃ = sin(θ₃)

    Note: The middle column has zeros because
    ω₂ (rotation about body Y) doesn't directly
    affect θ̇₁ or θ̇₂ in this parameterization.
    """
    ax3.text(0.02, 0.95, matrix_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # Panel 4: Why this is the answer
    ax4 = fig.add_subplot(234)
    ax4.axis('off')
    ax4.set_title('Why Option 1 is Correct', fontsize=11, fontweight='bold')

    why_text = """
    Checking the Options:

    ✗ Option 3: Has 1/cos(θ₂) factor
       → Wrong! Symmetric sequences have
         singularity at sin(θ₂)=0, not cos(θ₂)=0

    ✗ Option 2: Has θ₁ terms in matrix
       → Wrong! [B]⁻¹ should only depend on
         θ₂ and θ₃ (not θ₁) for this form

    ✓ Option 1: Has 1/sin(θ₂) factor and
       matrix depends only on θ₂, θ₃
       → Correct structure for (2-3-2)!

    Physical Insight:
    • When θ₂ → 0, axes 1 and 3 align
    • This creates gimbal lock
    • Hence sin(θ₂) in denominator
    """
    ax4.text(0.05, 0.95, why_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8))

    # Panel 5: Singularity visualization
    ax5 = fig.add_subplot(235, projection='3d')
    ax5.set_title('Gimbal Lock at θ₂=0°\n(Singularity Condition)', fontsize=10, fontweight='bold')

    # Show what happens when theta2 = 0
    theta1_s, theta2_s, theta3_s = np.radians([30, 0, 60])
    C_singular = Euler2322C(theta1_s, theta2_s, theta3_s)

    draw_frame(ax5, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.6, 1.5, 0.4)
    draw_frame(ax5, C_singular, 'B', ['red', 'green', 'blue'], 1.0, 2.5, 1.0)

    # Draw the Y-axes to show they're aligned
    ax5.plot([0, 0], [0, 1.3], [0, 0], 'g--', linewidth=2, alpha=0.7, label='Y-axes aligned!')

    setup_ax(ax5)
    ax5.view_init(elev=20, azim=45)
    ax5.legend(loc='upper left', fontsize=8)

    # Panel 6: Normal case
    ax6 = fig.add_subplot(236, projection='3d')
    ax6.set_title('Normal Case: θ₂=45°\n(No Singularity)', fontsize=10, fontweight='bold')

    theta1_n, theta2_n, theta3_n = np.radians([30, 45, 60])
    C_normal = Euler2322C(theta1_n, theta2_n, theta3_n)

    draw_frame(ax6, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.6, 1.5, 0.4)
    draw_frame(ax6, C_normal, 'B', ['red', 'green', 'blue'], 1.0, 2.5, 1.0)
    setup_ax(ax6)
    ax6.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('problem1_elaboration.png', dpi=150, bbox_inches='tight')
    print("  Saved: problem1_elaboration.png")
    plt.close()


# =============================================================================
# Problem 2: Euler Angle Integration Animation
# =============================================================================

def euler_321_kinematics(angles, t):
    """Differential kinematic equations for (3-2-1) Euler angles"""
    psi, theta, phi = angles

    # Angular velocity in body frame (deg/s -> rad/s)
    omega_deg = np.array([np.sin(0.1 * t), 0.01, np.cos(0.1 * t)]) * 20.0
    omega = np.radians(omega_deg)
    omega1, omega2, omega3 = omega

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    psi_dot = (sin_phi * omega2 + cos_phi * omega3) / cos_theta
    theta_dot = cos_phi * omega2 - sin_phi * omega3
    phi_dot = omega1 + (sin_phi * omega2 + cos_phi * omega3) * sin_theta / cos_theta

    return [psi_dot, theta_dot, phi_dot]


def create_problem2_animation():
    """Create animation for Problem 2: Euler angle integration"""
    print("Creating Problem 2 animation...")

    # Initial conditions
    psi_0, theta_0, phi_0 = np.radians([40, 30, 80])
    angles_0 = [psi_0, theta_0, phi_0]

    # Integrate
    dt = 0.1
    t = np.arange(0, 60 + dt, dt)
    solution = odeint(euler_321_kinematics, angles_0, t)

    # Create animation frames
    frames = []
    fig = plt.figure(figsize=(14, 6))

    # Sample every 1 second for animation (60 frames total)
    frame_indices = np.arange(0, len(t), 10)  # Every 1 second

    for idx in frame_indices:
        current_t = t[idx]
        psi, theta, phi = solution[idx]

        # Get angular velocity at this time
        omega_deg = np.array([np.sin(0.1 * current_t), 0.01, np.cos(0.1 * current_t)]) * 20.0
        omega = np.radians(omega_deg)

        # Compute DCM
        C = Euler3212C(psi, theta, phi)

        # Compute norm
        norm = np.sqrt(psi**2 + theta**2 + phi**2)

        # Left panel: 3D visualization
        ax1 = fig.add_subplot(121, projection='3d')

        # Highlight t=42s
        title_color = 'darkgreen' if abs(current_t - 42) < 0.5 else 'black'
        ax1.set_title(f't = {current_t:.1f}s\n'
                      f'ψ={np.degrees(psi):.1f}°, θ={np.degrees(theta):.1f}°, φ={np.degrees(phi):.1f}°',
                      fontsize=11, fontweight='bold', color=title_color)

        # Draw frames
        draw_frame(ax1, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.7, 1.5, 0.4)
        draw_frame(ax1, C, 'B', ['red', 'green', 'blue'], 1.0, 2.5, 1.0)

        # Draw angular velocity vector
        draw_angular_velocity(ax1, omega, C, scale=0.5, color='purple')

        setup_ax(ax1)
        ax1.view_init(elev=20, azim=30 + current_t * 0.5)

        # Right panel: Time history plots
        ax2 = fig.add_subplot(122)

        # Plot angles up to current time
        t_plot = t[:idx+1]
        sol_plot = solution[:idx+1]

        ax2.plot(t_plot, sol_plot[:, 0], 'b-', linewidth=2, label='ψ (yaw)')
        ax2.plot(t_plot, sol_plot[:, 1], 'g-', linewidth=2, label='θ (pitch)')
        ax2.plot(t_plot, sol_plot[:, 2], 'r-', linewidth=2, label='φ (roll)')

        # Mark current point
        ax2.plot(current_t, psi, 'bo', markersize=8)
        ax2.plot(current_t, theta, 'go', markersize=8)
        ax2.plot(current_t, phi, 'ro', markersize=8)

        # Mark t=42s line
        ax2.axvline(x=42, color='gray', linestyle='--', alpha=0.5, label='t=42s')

        ax2.set_xlim([0, 60])
        ax2.set_ylim([min(solution.min() - 0.5, -2), max(solution.max() + 0.5, 12)])
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Angle (rad)', fontsize=10)
        ax2.set_title(f'Euler Angles vs Time\nNorm at t={current_t:.1f}s: {norm:.4f} rad',
                      fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_frame(fig, frames)
        ax1.clear()
        ax2.clear()

    plt.close(fig)

    # Save animation
    frames[0].save('problem2_euler_integration.gif', save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    print(f"  Saved: problem2_euler_integration.gif ({len(frames)} frames)")

    # Also create a summary plot
    create_problem2_summary(t, solution)


def create_problem2_summary(t, solution):
    """Create a summary plot for Problem 2"""
    print("Creating Problem 2 summary plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Problem 2: Euler Angle Numerical Integration Summary',
                 fontsize=14, fontweight='bold')

    # Panel 1: All angles vs time
    ax1 = axes[0, 0]
    ax1.plot(t, np.degrees(solution[:, 0]), 'b-', linewidth=2, label='ψ (yaw)')
    ax1.plot(t, np.degrees(solution[:, 1]), 'g-', linewidth=2, label='θ (pitch)')
    ax1.plot(t, np.degrees(solution[:, 2]), 'r-', linewidth=2, label='φ (roll)')
    ax1.axvline(x=42, color='gray', linestyle='--', linewidth=2, label='t=42s')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Euler Angles vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Euler angle norm
    ax2 = axes[0, 1]
    norms = np.sqrt(solution[:, 0]**2 + solution[:, 1]**2 + solution[:, 2]**2)
    ax2.plot(t, norms, 'purple', linewidth=2)
    ax2.axvline(x=42, color='gray', linestyle='--', linewidth=2)

    # Mark t=42s
    idx_42 = np.argmin(np.abs(t - 42))
    norm_42 = norms[idx_42]
    ax2.plot(42, norm_42, 'ro', markersize=12, label=f'At t=42s: {norm_42:.6f} rad')
    ax2.annotate(f'{norm_42:.4f} rad', xy=(42, norm_42), xytext=(48, norm_42 + 1),
                 fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red'))

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Norm (rad)')
    ax2.set_title('Euler Angle Norm √(ψ² + θ² + φ²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Angular velocity
    ax3 = axes[1, 0]
    omega1 = 20 * np.sin(0.1 * t)
    omega2 = 20 * 0.01 * np.ones_like(t)
    omega3 = 20 * np.cos(0.1 * t)
    ax3.plot(t, omega1, 'r-', linewidth=2, label='ω₁ = 20·sin(0.1t)')
    ax3.plot(t, omega2, 'g-', linewidth=2, label='ω₂ = 0.2')
    ax3.plot(t, omega3, 'b-', linewidth=2, label='ω₃ = 20·cos(0.1t)')
    ax3.axvline(x=42, color='gray', linestyle='--', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angular Velocity (deg/s)')
    ax3.set_title('Body Angular Velocity Components')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Problem summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    PROBLEM 2 SUMMARY
    ═══════════════════════════════════════════════

    Initial Conditions (t=0):
    ─────────────────────────
    • ψ₀ (yaw)   = 40° = 0.6981 rad
    • θ₀ (pitch) = 30° = 0.5236 rad
    • φ₀ (roll)  = 80° = 1.3963 rad

    Angular Velocity (Body Frame):
    ──────────────────────────────
    • ω₁ = 20·sin(0.1t) deg/s
    • ω₂ = 0.2 deg/s
    • ω₃ = 20·cos(0.1t) deg/s

    ═══════════════════════════════════════════════
    RESULT AT t = 42 seconds:
    ═══════════════════════════════════════════════

    • ψ = {solution[idx_42, 0]:.6f} rad ({np.degrees(solution[idx_42, 0]):.2f}°)
    • θ = {solution[idx_42, 1]:.6f} rad ({np.degrees(solution[idx_42, 1]):.2f}°)
    • φ = {solution[idx_42, 2]:.6f} rad ({np.degrees(solution[idx_42, 2]):.2f}°)

    ┌─────────────────────────────────────────────┐
    │  NORM = √(ψ² + θ² + φ²) = {norm_42:.6f} rad  │
    └─────────────────────────────────────────────┘
    """
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('problem2_summary.png', dpi=150, bbox_inches='tight')
    print("  Saved: problem2_summary.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Creating Visualizations for Problems 1 and 2")
    print("=" * 60 + "\n")

    elaborate_problem1()
    print()
    create_problem2_animation()

    print("\n" + "=" * 60)
    print("Done! Generated files:")
    print("  - problem1_elaboration.png     (Problem 1 explanation)")
    print("  - problem2_euler_integration.gif (Problem 2 animation)")
    print("  - problem2_summary.png         (Problem 2 summary)")
    print("=" * 60)


if __name__ == "__main__":
    main()
