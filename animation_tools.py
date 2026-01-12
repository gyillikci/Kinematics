#!/usr/bin/env python3
"""
Fast Animated Euler Angle Visualizations
Lightweight version with fewer frames for quick generation.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io


def Rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def Ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def Rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

def Euler3212C(angles_deg):
    psi, theta, phi = [math.radians(a) for a in angles_deg]
    return Rx(phi) @ Ry(theta) @ Rz(psi)

def Euler3132C(angles_deg):
    alpha, beta, gamma = [math.radians(a) for a in angles_deg]
    return Rz(gamma) @ Rx(beta) @ Rz(alpha)


def draw_frame(ax, dcm, label, colors=['red', 'green', 'blue'], length=1.0, lw=2.5, alpha=1.0):
    """Draw coordinate frame axes."""
    origin = [0, 0, 0]
    for i, (c, axis_name) in enumerate(zip(colors, ['x', 'y', 'z'])):
        axis = dcm.T[:, i] * length
        ax.quiver(0, 0, 0, axis[0], axis[1], axis[2],
                  color=c, linewidth=lw, arrow_length_ratio=0.12, alpha=alpha)
        tip = axis * 1.2
        ax.text(tip[0], tip[1], tip[2], f'{label}_{axis_name}',
                color=c, fontsize=9, fontweight='bold', alpha=alpha)


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


def create_321_animation():
    """Animate (3-2-1) rotation sequence."""
    print("Creating (3-2-1) rotation animation...")

    angles = [10, 20, 30]  # psi, theta, phi in degrees
    angles_rad = [math.radians(a) for a in angles]

    frames = []
    fig = plt.figure(figsize=(8, 7))

    n_frames = 15  # frames per rotation

    # Phase 1: Rotate about Z (psi)
    for i in range(n_frames):
        ax = fig.add_subplot(111, projection='3d')
        t = i / (n_frames - 1)
        angle = t * angles_rad[0]
        C = Rz(angle)

        ax.set_title(f'Step 1: Rotate about Z-axis\nψ = {math.degrees(angle):.1f}° → {angles[0]}°',
                     fontsize=13, fontweight='bold')
        draw_frame(ax, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.7, 1.5, 0.4)
        draw_frame(ax, C, 'B')
        setup_ax(ax)
        ax.view_init(elev=25, azim=45)
        save_frame(fig, frames)
        ax.clear()

    C1 = Rz(angles_rad[0])

    # Phase 2: Rotate about Y' (theta)
    for i in range(n_frames):
        ax = fig.add_subplot(111, projection='3d')
        t = i / (n_frames - 1)
        angle = t * angles_rad[1]
        C = Ry(angle) @ C1

        ax.set_title(f'Step 2: Rotate about Y\'-axis\nθ = {math.degrees(angle):.1f}° → {angles[1]}°',
                     fontsize=13, fontweight='bold')
        draw_frame(ax, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.6, 1, 0.3)
        draw_frame(ax, C1, "N'", ['pink', 'palegreen', 'powderblue'], 0.7, 1.5, 0.5)
        draw_frame(ax, C, 'B')
        setup_ax(ax)
        ax.view_init(elev=25, azim=45)
        save_frame(fig, frames)
        ax.clear()

    C2 = Ry(angles_rad[1]) @ C1

    # Phase 3: Rotate about X'' (phi)
    for i in range(n_frames):
        ax = fig.add_subplot(111, projection='3d')
        t = i / (n_frames - 1)
        angle = t * angles_rad[2]
        C = Rx(angle) @ C2

        ax.set_title(f'Step 3: Rotate about X\'\'-axis\nφ = {math.degrees(angle):.1f}° → {angles[2]}°',
                     fontsize=13, fontweight='bold')
        draw_frame(ax, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.5, 1, 0.2)
        draw_frame(ax, C2, "N''", ['pink', 'palegreen', 'powderblue'], 0.7, 1.5, 0.5)
        draw_frame(ax, C, 'B')
        setup_ax(ax)
        ax.view_init(elev=25, azim=45)
        save_frame(fig, frames)
        ax.clear()

    C_final = Rx(angles_rad[2]) @ C2

    # Phase 4: Final - rotate view
    for i in range(20):
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'FINAL: (3-2-1) = ({angles[0]}°, {angles[1]}°, {angles[2]}°)',
                     fontsize=13, fontweight='bold', color='darkgreen')
        draw_frame(ax, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.7, 1.5, 0.5)
        draw_frame(ax, C_final, 'B')
        setup_ax(ax)
        ax.view_init(elev=25, azim=45 + i * 6)
        save_frame(fig, frames)
        ax.clear()

    plt.close(fig)

    frames[0].save('rotation_321_animation.gif', save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    print(f"  Saved: rotation_321_animation.gif ({len(frames)} frames)")


def create_comparison_animation():
    """Animate (3-2-1) vs (3-1-3) comparison."""
    print("Creating (3-2-1) vs (3-1-3) comparison animation...")

    euler_321 = [10, 20, 30]
    euler_313 = [40.642342, 35.531348, -36.052389]

    angles_321_rad = [math.radians(a) for a in euler_321]
    angles_313_rad = [math.radians(a) for a in euler_313]

    frames = []
    fig = plt.figure(figsize=(12, 5))

    # Animate both simultaneously
    for i in range(25):
        t = i / 24

        # (3-2-1)
        C_321 = Rx(t*angles_321_rad[2]) @ Ry(t*angles_321_rad[1]) @ Rz(t*angles_321_rad[0])
        # (3-1-3)
        C_313 = Rz(t*angles_313_rad[2]) @ Rx(t*angles_313_rad[1]) @ Rz(t*angles_313_rad[0])

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title(f'(3-2-1): ψ={t*euler_321[0]:.0f}°, θ={t*euler_321[1]:.0f}°, φ={t*euler_321[2]:.0f}°',
                      fontsize=11, fontweight='bold')
        draw_frame(ax1, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.7, 1.5, 0.4)
        draw_frame(ax1, C_321, 'B', ['red', 'green', 'blue'])
        setup_ax(ax1)
        ax1.view_init(elev=25, azim=45)

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title(f'(3-1-3): α={t*euler_313[0]:.0f}°, β={t*euler_313[1]:.0f}°, γ={t*euler_313[2]:.0f}°',
                      fontsize=11, fontweight='bold')
        draw_frame(ax2, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.7, 1.5, 0.4)
        draw_frame(ax2, C_313, 'B', ['darkred', 'darkgreen', 'darkblue'])
        setup_ax(ax2)
        ax2.view_init(elev=25, azim=45)

        plt.tight_layout()
        save_frame(fig, frames)
        ax1.clear()
        ax2.clear()

    # Final rotating view
    C_321_final = Euler3212C(euler_321)
    C_313_final = Euler3132C(euler_313)

    for i in range(20):
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title(f'(3-2-1): ({euler_321[0]}°, {euler_321[1]}°, {euler_321[2]}°)\nSAME RESULT!',
                      fontsize=11, fontweight='bold', color='green')
        draw_frame(ax1, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.7, 1.5, 0.4)
        draw_frame(ax1, C_321_final, 'B', ['red', 'green', 'blue'])
        setup_ax(ax1)
        ax1.view_init(elev=25, azim=45 + i * 6)

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title(f'(3-1-3): ({euler_313[0]:.1f}°, {euler_313[1]:.1f}°, {euler_313[2]:.1f}°)\nSAME RESULT!',
                      fontsize=11, fontweight='bold', color='green')
        draw_frame(ax2, np.eye(3), 'N', ['lightcoral', 'lightgreen', 'lightblue'], 0.7, 1.5, 0.4)
        draw_frame(ax2, C_313_final, 'B', ['darkred', 'darkgreen', 'darkblue'])
        setup_ax(ax2)
        ax2.view_init(elev=25, azim=45 + i * 6)

        plt.tight_layout()
        save_frame(fig, frames)
        ax1.clear()
        ax2.clear()

    plt.close(fig)

    frames[0].save('euler_321_vs_313_animation.gif', save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    print(f"  Saved: euler_321_vs_313_animation.gif ({len(frames)} frames)")


def create_relative_attitude_animation():
    """Animate relative attitude calculation."""
    print("Creating relative attitude animation...")

    BN_euler = [10, 20, 30]
    RN_euler = [-5, 5, 5]
    BR_euler = [13.223818, 16.368343, 23.617628]

    C_BN = Euler3212C(BN_euler)
    C_RN = Euler3212C(RN_euler)
    C_BR = Euler3212C(BR_euler)

    frames = []
    fig = plt.figure(figsize=(8, 7))

    # Phase 1: Show N, then add B
    for i in range(12):
        ax = fig.add_subplot(111, projection='3d')
        t = min(i / 8, 1.0)
        ax.set_title(f'Step 1: B relative to N\n(3-2-1): ({BN_euler[0]}°, {BN_euler[1]}°, {BN_euler[2]}°)',
                     fontsize=12, fontweight='bold')
        draw_frame(ax, np.eye(3), 'N', ['gray', 'gray', 'gray'], 0.7, 1.5, 0.5)
        if t > 0:
            draw_frame(ax, C_BN, 'B', ['red', 'green', 'blue'], 1.0, 2.5, t)
        setup_ax(ax)
        ax.view_init(elev=25, azim=30 + i * 2)
        save_frame(fig, frames)
        ax.clear()

    # Phase 2: Add R frame
    for i in range(12):
        ax = fig.add_subplot(111, projection='3d')
        t = min(i / 8, 1.0)
        ax.set_title(f'Step 2: R relative to N\n(3-2-1): ({RN_euler[0]}°, {RN_euler[1]}°, {RN_euler[2]}°)',
                     fontsize=12, fontweight='bold')
        draw_frame(ax, np.eye(3), 'N', ['gray', 'gray', 'gray'], 0.6, 1, 0.4)
        draw_frame(ax, C_BN, 'B', ['red', 'green', 'blue'], 1.0, 2.5, 1.0)
        if t > 0:
            draw_frame(ax, C_RN, 'R', ['orange', 'lime', 'cyan'], 1.0, 2.5, t)
        setup_ax(ax)
        ax.view_init(elev=25, azim=54 + i * 2)
        save_frame(fig, frames)
        ax.clear()

    # Phase 3: Show all frames
    for i in range(15):
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Step 3: Computing [BR] = [BN][RN]ᵀ',
                     fontsize=12, fontweight='bold')
        draw_frame(ax, np.eye(3), 'N', ['gray', 'gray', 'gray'], 0.5, 1, 0.3)
        draw_frame(ax, C_BN, 'B', ['red', 'green', 'blue'], 1.0, 2.5)
        draw_frame(ax, C_RN, 'R', ['orange', 'lime', 'cyan'], 1.0, 2.5)
        setup_ax(ax)
        ax.view_init(elev=25, azim=78 + i * 3)
        save_frame(fig, frames)
        ax.clear()

    # Phase 4: Show result - B relative to R
    for i in range(20):
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'RESULT: B relative to R\n(3-2-1): ({BR_euler[0]:.2f}°, {BR_euler[1]:.2f}°, {BR_euler[2]:.2f}°)',
                     fontsize=12, fontweight='bold', color='darkgreen')
        draw_frame(ax, np.eye(3), 'R', ['moccasin', 'palegreen', 'lightcyan'], 0.8, 1.5, 0.6)
        draw_frame(ax, C_BR, 'B', ['red', 'green', 'blue'], 1.0, 2.5)
        setup_ax(ax)
        ax.view_init(elev=25, azim=123 + i * 6)
        save_frame(fig, frames)
        ax.clear()

    plt.close(fig)

    frames[0].save('relative_attitude_animation.gif', save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    print(f"  Saved: relative_attitude_animation.gif ({len(frames)} frames)")


def main():
    print("=" * 50)
    print("Generating Euler Angle Animations")
    print("=" * 50 + "\n")

    create_321_animation()
    create_comparison_animation()
    create_relative_attitude_animation()

    print("\n" + "=" * 50)
    print("Done! Generated files:")
    print("  - rotation_321_animation.gif")
    print("  - euler_321_vs_313_animation.gif")
    print("  - relative_attitude_animation.gif")
    print("=" * 50)


if __name__ == "__main__":
    main()
