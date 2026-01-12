#!/usr/bin/env python3
"""
Euler Angle Visualization Tools

Provides 3D visualizations to understand:
1. Coordinate frame rotations
2. (3-2-1) to (3-1-3) Euler angle conversion
3. Relative attitude calculations
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================================
# Rotation Matrix Functions
# ============================================================================

def Rx(angle):
    """Elementary rotation about X-axis (1-axis)"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, s],
                     [0, -s, c]])


def Ry(angle):
    """Elementary rotation about Y-axis (2-axis)"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, -s],
                     [0, 1, 0],
                     [s, 0, c]])


def Rz(angle):
    """Elementary rotation about Z-axis (3-axis)"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s, 0],
                     [-s, c, 0],
                     [0, 0, 1]])


def Euler3212C(angles_deg):
    """Convert (3-2-1) Euler angles to DCM. Input in degrees."""
    psi, theta, phi = [math.radians(a) for a in angles_deg]
    return Rx(phi) @ Ry(theta) @ Rz(psi)


def Euler3132C(angles_deg):
    """Convert (3-1-3) Euler angles to DCM. Input in degrees."""
    alpha, beta, gamma = [math.radians(a) for a in angles_deg]
    return Rz(gamma) @ Rx(beta) @ Rz(alpha)


def C2Euler321(C):
    """Extract (3-2-1) Euler angles from DCM. Output in degrees."""
    psi = math.atan2(C[0, 1], C[0, 0])
    theta = math.asin(-C[0, 2])
    phi = math.atan2(C[1, 2], C[2, 2])
    return [math.degrees(psi), math.degrees(theta), math.degrees(phi)]


def C2Euler313(C):
    """Extract (3-1-3) Euler angles from DCM. Output in degrees."""
    alpha = math.atan2(C[2, 0], -C[2, 1])
    beta = math.acos(C[2, 2])
    gamma = math.atan2(C[0, 2], C[1, 2])
    return [math.degrees(alpha), math.degrees(beta), math.degrees(gamma)]


# ============================================================================
# Visualization Functions
# ============================================================================

def draw_frame(ax, origin, dcm, label, colors=None, length=1.0, linewidth=2):
    """
    Draw a 3D coordinate frame.

    Parameters:
    - ax: matplotlib 3D axis
    - origin: [x, y, z] position of frame origin
    - dcm: 3x3 Direction Cosine Matrix (columns are frame axes in reference frame)
    - label: Frame label (e.g., 'N', 'B', 'R')
    - colors: List of colors for [x, y, z] axes
    - length: Length of axis arrows
    """
    if colors is None:
        colors = ['red', 'green', 'blue']

    origin = np.array(origin)

    # For visualization, we plot the frame axes as they appear in the inertial frame
    # The DCM rows represent the body frame axes expressed in the inertial frame
    axes_labels = ['x', 'y', 'z']

    for i in range(3):
        # DCM.T columns give the body axes in inertial coordinates
        axis_dir = dcm.T[:, i] * length
        ax.quiver(origin[0], origin[1], origin[2],
                  axis_dir[0], axis_dir[1], axis_dir[2],
                  color=colors[i], linewidth=linewidth, arrow_length_ratio=0.1)

        # Label the axis tip
        tip = origin + axis_dir * 1.15
        ax.text(tip[0], tip[1], tip[2], f'{label}_{axes_labels[i]}',
                color=colors[i], fontsize=9, fontweight='bold')


def draw_rotation_sequence(ax, angles_deg, sequence='321', origin=[0,0,0],
                           show_intermediate=True):
    """
    Visualize a rotation sequence step by step.

    Parameters:
    - angles_deg: List of 3 angles in degrees
    - sequence: '321' for (3-2-1) or '313' for (3-1-3)
    """
    origin = np.array(origin)

    # Identity (reference frame)
    C = np.eye(3)

    if sequence == '321':
        rotations = [
            (Rz(math.radians(angles_deg[0])), f'R3({angles_deg[0]}°)', 'After Z rotation'),
            (Ry(math.radians(angles_deg[1])), f'R2({angles_deg[1]}°)', 'After Y rotation'),
            (Rx(math.radians(angles_deg[2])), f'R1({angles_deg[2]}°)', 'After X rotation')
        ]
    else:  # 313
        rotations = [
            (Rz(math.radians(angles_deg[0])), f'R3({angles_deg[0]}°)', 'After 1st Z rotation'),
            (Rx(math.radians(angles_deg[1])), f'R1({angles_deg[1]}°)', 'After X rotation'),
            (Rz(math.radians(angles_deg[2])), f'R3({angles_deg[2]}°)', 'After 2nd Z rotation')
        ]

    # Draw reference frame (dashed)
    draw_frame(ax, origin, np.eye(3), 'N',
               colors=['lightcoral', 'lightgreen', 'lightblue'],
               length=0.8, linewidth=1)

    # Apply rotations sequentially
    intermediate_frames = []
    for i, (R, name, desc) in enumerate(rotations):
        C = R @ C
        intermediate_frames.append((C.copy(), desc))

    if show_intermediate:
        # Show intermediate frames with transparency effect
        alphas = [0.3, 0.5, 1.0]
        for i, (C_int, desc) in enumerate(intermediate_frames):
            if i < len(intermediate_frames) - 1:
                draw_frame(ax, origin, C_int, f'Step{i+1}',
                          colors=[f'C{i}', f'C{i}', f'C{i}'],
                          length=0.6, linewidth=1)

    # Draw final frame
    draw_frame(ax, origin, C, 'B', length=1.0, linewidth=2.5)

    return C


def visualize_321_to_313_conversion():
    """
    Visualize the conversion from (3-2-1) to (3-1-3) Euler angles.
    Shows that both representations produce the same final orientation.
    """
    fig = plt.figure(figsize=(16, 6))

    # Given angles
    euler_321 = [10, 20, 30]  # degrees
    euler_313 = [40.642342, 35.531348, -36.052389]  # equivalent (3-1-3)

    # Subplot 1: (3-2-1) rotation sequence
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title(f'(3-2-1) Euler Angles\n({euler_321[0]}°, {euler_321[1]}°, {euler_321[2]}°)',
                  fontsize=12, fontweight='bold')

    C_321 = Euler3212C(euler_321)
    draw_frame(ax1, [0,0,0], np.eye(3), 'N',
               colors=['lightcoral', 'lightgreen', 'lightblue'], length=0.7, linewidth=1)
    draw_frame(ax1, [0,0,0], C_321, 'B', length=1.0, linewidth=2.5)

    setup_3d_axis(ax1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Subplot 2: (3-1-3) rotation sequence
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(f'(3-1-3) Euler Angles\n({euler_313[0]:.2f}°, {euler_313[1]:.2f}°, {euler_313[2]:.2f}°)',
                  fontsize=12, fontweight='bold')

    C_313 = Euler3132C(euler_313)
    draw_frame(ax2, [0,0,0], np.eye(3), 'N',
               colors=['lightcoral', 'lightgreen', 'lightblue'], length=0.7, linewidth=1)
    draw_frame(ax2, [0,0,0], C_313, 'B', length=1.0, linewidth=2.5)

    setup_3d_axis(ax2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Subplot 3: Overlay comparison
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Overlay Comparison\n(Both should match)', fontsize=12, fontweight='bold')

    draw_frame(ax3, [0,0,0], np.eye(3), 'N',
               colors=['lightcoral', 'lightgreen', 'lightblue'], length=0.7, linewidth=1)
    draw_frame(ax3, [0,0,0], C_321, '321',
               colors=['red', 'green', 'blue'], length=1.0, linewidth=2.5)
    draw_frame(ax3, [0,0,0], C_313, '313',
               colors=['darkred', 'darkgreen', 'darkblue'], length=0.9, linewidth=1.5)

    setup_3d_axis(ax3)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Add text showing the conversion
    fig.text(0.5, 0.02,
             f'Conversion: (3-2-1) [{euler_321[0]}°, {euler_321[1]}°, {euler_321[2]}°] → '
             f'(3-1-3) [{euler_313[0]:.2f}°, {euler_313[1]:.2f}°, {euler_313[2]:.2f}°]',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('euler_321_to_313_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: euler_321_to_313_visualization.png")
    plt.close()


def visualize_relative_attitude():
    """
    Visualize the relative attitude calculation.
    B relative to N, R relative to N, and B relative to R.
    """
    fig = plt.figure(figsize=(16, 10))

    # Given angles
    BN_euler = [10, 20, 30]    # B relative to N
    RN_euler = [-5, 5, 5]      # R relative to N
    BR_euler = [13.223818, 16.368343, 23.617628]  # B relative to R (result)

    # Compute DCMs
    C_BN = Euler3212C(BN_euler)
    C_RN = Euler3212C(RN_euler)
    C_BR = Euler3212C(BR_euler)

    # Subplot 1: All frames relative to N
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_title('All Frames in Inertial (N) Coordinates', fontsize=12, fontweight='bold')

    # Draw N frame
    draw_frame(ax1, [0,0,0], np.eye(3), 'N',
               colors=['gray', 'gray', 'gray'], length=0.6, linewidth=1)

    # Draw B frame (relative to N)
    draw_frame(ax1, [0,0,0], C_BN, 'B',
               colors=['red', 'green', 'blue'], length=1.0, linewidth=2.5)

    # Draw R frame (relative to N)
    draw_frame(ax1, [0,0,0], C_RN, 'R',
               colors=['orange', 'lime', 'cyan'], length=1.0, linewidth=2.5)

    setup_3d_axis(ax1)
    ax1.set_xlabel('X_N')
    ax1.set_ylabel('Y_N')
    ax1.set_zlabel('Z_N')

    # Subplot 2: B relative to N
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.set_title(f'B relative to N\n(3-2-1): ({BN_euler[0]}°, {BN_euler[1]}°, {BN_euler[2]}°)',
                  fontsize=11, fontweight='bold')

    draw_frame(ax2, [0,0,0], np.eye(3), 'N',
               colors=['lightcoral', 'lightgreen', 'lightblue'], length=0.7, linewidth=1)
    draw_frame(ax2, [0,0,0], C_BN, 'B',
               colors=['red', 'green', 'blue'], length=1.0, linewidth=2.5)

    setup_3d_axis(ax2)

    # Subplot 3: R relative to N
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.set_title(f'R relative to N\n(3-2-1): ({RN_euler[0]}°, {RN_euler[1]}°, {RN_euler[2]}°)',
                  fontsize=11, fontweight='bold')

    draw_frame(ax3, [0,0,0], np.eye(3), 'N',
               colors=['lightcoral', 'lightgreen', 'lightblue'], length=0.7, linewidth=1)
    draw_frame(ax3, [0,0,0], C_RN, 'R',
               colors=['orange', 'lime', 'cyan'], length=1.0, linewidth=2.5)

    setup_3d_axis(ax3)

    # Subplot 4: B relative to R
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title(f'B relative to R (Result)\n(3-2-1): ({BR_euler[0]:.2f}°, {BR_euler[1]:.2f}°, {BR_euler[2]:.2f}°)',
                  fontsize=11, fontweight='bold')

    # Here we show R as the reference frame
    draw_frame(ax4, [0,0,0], np.eye(3), 'R',
               colors=['moccasin', 'palegreen', 'lightcyan'], length=0.7, linewidth=1)
    draw_frame(ax4, [0,0,0], C_BR, 'B',
               colors=['red', 'green', 'blue'], length=1.0, linewidth=2.5)

    setup_3d_axis(ax4)

    # Add explanation text
    fig.text(0.5, 0.02,
             '[BR] = [BN][RN]ᵀ  →  B relative to R computed by composing rotations',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('relative_attitude_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: relative_attitude_visualization.png")
    plt.close()


def visualize_rotation_sequence_321():
    """
    Visualize the step-by-step (3-2-1) rotation sequence.
    """
    fig = plt.figure(figsize=(16, 5))

    angles = [10, 20, 30]  # psi, theta, phi

    # Step 0: Initial (N frame)
    ax0 = fig.add_subplot(141, projection='3d')
    ax0.set_title('Step 0: Initial\n(N Frame)', fontsize=11, fontweight='bold')
    draw_frame(ax0, [0,0,0], np.eye(3), 'N', length=1.0, linewidth=2.5)
    setup_3d_axis(ax0)

    # Step 1: After R3 (Z rotation by psi)
    ax1 = fig.add_subplot(142, projection='3d')
    ax1.set_title(f'Step 1: R₃({angles[0]}°)\nRotate about Z', fontsize=11, fontweight='bold')
    C1 = Rz(math.radians(angles[0]))
    draw_frame(ax1, [0,0,0], np.eye(3), 'N',
               colors=['lightcoral', 'lightgreen', 'lightblue'], length=0.7, linewidth=1)
    draw_frame(ax1, [0,0,0], C1, "N'", length=1.0, linewidth=2.5)
    # Draw rotation arc
    draw_rotation_arc(ax1, 'z', angles[0], color='blue')
    setup_3d_axis(ax1)

    # Step 2: After R2 (Y rotation by theta)
    ax2 = fig.add_subplot(143, projection='3d')
    ax2.set_title(f'Step 2: R₂({angles[1]}°)\nRotate about Y\'', fontsize=11, fontweight='bold')
    C2 = Ry(math.radians(angles[1])) @ C1
    draw_frame(ax2, [0,0,0], np.eye(3), 'N',
               colors=['lightcoral', 'lightgreen', 'lightblue'], length=0.6, linewidth=1)
    draw_frame(ax2, [0,0,0], C1, "N'",
               colors=['pink', 'palegreen', 'powderblue'], length=0.7, linewidth=1)
    draw_frame(ax2, [0,0,0], C2, "N''", length=1.0, linewidth=2.5)
    setup_3d_axis(ax2)

    # Step 3: After R1 (X rotation by phi) - Final B frame
    ax3 = fig.add_subplot(144, projection='3d')
    ax3.set_title(f'Step 3: R₁({angles[2]}°)\nRotate about X\'\' → B', fontsize=11, fontweight='bold')
    C3 = Rx(math.radians(angles[2])) @ C2
    draw_frame(ax3, [0,0,0], np.eye(3), 'N',
               colors=['lightcoral', 'lightgreen', 'lightblue'], length=0.6, linewidth=1)
    draw_frame(ax3, [0,0,0], C3, 'B', length=1.0, linewidth=2.5)
    setup_3d_axis(ax3)

    fig.text(0.5, 0.02,
             f'(3-2-1) Sequence: R₁({angles[2]}°) · R₂({angles[1]}°) · R₃({angles[0]}°) = [BN]',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('rotation_sequence_321.png', dpi=150, bbox_inches='tight')
    print("Saved: rotation_sequence_321.png")
    plt.close()


def draw_rotation_arc(ax, axis, angle_deg, radius=0.5, color='purple'):
    """Draw an arc showing the rotation direction."""
    angle_rad = math.radians(angle_deg)
    t = np.linspace(0, angle_rad, 30)

    if axis == 'z':
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.zeros_like(t)
    elif axis == 'y':
        x = radius * np.sin(t)
        y = np.zeros_like(t)
        z = radius * np.cos(t)
    else:  # x
        x = np.zeros_like(t)
        y = radius * np.cos(t)
        z = radius * np.sin(t)

    ax.plot(x, y, z, color=color, linewidth=2, linestyle='--')


def setup_3d_axis(ax, limit=1.3):
    """Configure 3D axis properties."""
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)


def create_summary_diagram():
    """
    Create a summary diagram showing both problems and solutions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Problem 1: (3-2-1) to (3-1-3) conversion
    ax1 = axes[0, 0]
    ax1.axis('off')
    ax1.set_title('Problem 1: Euler Angle Conversion', fontsize=14, fontweight='bold', pad=20)

    text1 = """
    Given: (3-2-1) Euler Angles = (10°, 20°, 30°)
    Find:  Equivalent (3-1-3) Euler Angles

    Method:
    1. Convert (3-2-1) → DCM using [C] = R₁(φ)R₂(θ)R₃(ψ)
    2. Extract (3-1-3) angles from DCM:
       α = atan2(C₃₁, -C₃₂)
       β = acos(C₃₃)
       γ = atan2(C₁₃, C₂₃)

    ═══════════════════════════════════════
    RESULT: (3-1-3) = (40.64°, 35.53°, -36.05°)
    ═══════════════════════════════════════
    """
    ax1.text(0.1, 0.5, text1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # Problem 2: Relative Attitude
    ax2 = axes[0, 1]
    ax2.axis('off')
    ax2.set_title('Problem 2: Relative Attitude', fontsize=14, fontweight='bold', pad=20)

    text2 = """
    Given: B/N (3-2-1) = (10°, 20°, 30°)
           R/N (3-2-1) = (-5°, 5°, 5°)
    Find:  B/R in (3-2-1) Euler Angles

    Method:
    1. Compute [BN] from B/N Euler angles
    2. Compute [RN] from R/N Euler angles
    3. [BR] = [BN][NR] = [BN][RN]ᵀ
    4. Extract (3-2-1) angles from [BR]

    ═══════════════════════════════════════
    RESULT: B/R = (13.22°, 16.37°, 23.62°)
    ═══════════════════════════════════════
    """
    ax2.text(0.1, 0.5, text2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # DCM for Problem 1
    ax3 = axes[1, 0]
    ax3.axis('off')
    ax3.set_title('DCM from (3-2-1) angles (10°, 20°, 30°)', fontsize=12, fontweight='bold')

    C = Euler3212C([10, 20, 30])
    dcm_text = f"""
         ┌                                        ┐
         │  {C[0,0]:9.6f}   {C[0,1]:9.6f}   {C[0,2]:9.6f}  │
    [C] =│  {C[1,0]:9.6f}   {C[1,1]:9.6f}   {C[1,2]:9.6f}  │
         │  {C[2,0]:9.6f}   {C[2,1]:9.6f}   {C[2,2]:9.6f}  │
         └                                        ┘

    Verification: det([C]) = {np.linalg.det(C):.10f}
                  [C][C]ᵀ = I (orthogonal)
    """
    ax3.text(0.1, 0.5, dcm_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    # DCM for Problem 2
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.set_title('DCM [BR] = B relative to R', fontsize=12, fontweight='bold')

    C_BN = Euler3212C([10, 20, 30])
    C_RN = Euler3212C([-5, 5, 5])
    C_BR = C_BN @ C_RN.T

    dcm_text2 = f"""
          ┌                                        ┐
          │  {C_BR[0,0]:9.6f}   {C_BR[0,1]:9.6f}   {C_BR[0,2]:9.6f}  │
    [BR] =│  {C_BR[1,0]:9.6f}   {C_BR[1,1]:9.6f}   {C_BR[1,2]:9.6f}  │
          │  {C_BR[2,0]:9.6f}   {C_BR[2,1]:9.6f}   {C_BR[2,2]:9.6f}  │
          └                                        ┘

    Extracted (3-2-1) angles:
      ψ = {C2Euler321(C_BR)[0]:.6f}°
      θ = {C2Euler321(C_BR)[1]:.6f}°
      φ = {C2Euler321(C_BR)[2]:.6f}°
    """
    ax4.text(0.1, 0.5, dcm_text2, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8))

    plt.tight_layout()
    plt.savefig('euler_angles_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: euler_angles_summary.png")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Euler Angle Visualizations")
    print("=" * 60)

    print("\n1. Creating (3-2-1) to (3-1-3) conversion visualization...")
    visualize_321_to_313_conversion()

    print("\n2. Creating relative attitude visualization...")
    visualize_relative_attitude()

    print("\n3. Creating (3-2-1) rotation sequence visualization...")
    visualize_rotation_sequence_321()

    print("\n4. Creating summary diagram...")
    create_summary_diagram()

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - euler_321_to_313_visualization.png")
    print("  - relative_attitude_visualization.png")
    print("  - rotation_sequence_321.png")
    print("  - euler_angles_summary.png")


if __name__ == "__main__":
    main()
