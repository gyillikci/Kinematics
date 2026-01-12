#!/usr/bin/env python3
"""
Euler Angle Conversion: (3-2-1) to (3-1-3)

This script converts Euler angles from (3-2-1) sequence to (3-1-3) sequence.
Given: (3-2-1) Euler angles = (10, 20, 30) degrees
Find: Equivalent (3-1-3) Euler angles

Method:
1. Convert (3-2-1) Euler angles to Direction Cosine Matrix (DCM)
2. Extract (3-1-3) Euler angles from the DCM
"""

import numpy as np
import math


def Euler3212C(q):
    """
    Convert (3-2-1) Euler angles to Direction Cosine Matrix.

    Input: q = [psi, theta, phi] in radians
           psi   = rotation about 3-axis (Z)
           theta = rotation about 2-axis (Y)
           phi   = rotation about 1-axis (X)

    Output: 3x3 Direction Cosine Matrix C
    """
    st1 = math.sin(q[0])  # sin(psi)
    ct1 = math.cos(q[0])  # cos(psi)
    st2 = math.sin(q[1])  # sin(theta)
    ct2 = math.cos(q[1])  # cos(theta)
    st3 = math.sin(q[2])  # sin(phi)
    ct3 = math.cos(q[2])  # cos(phi)

    C = np.zeros((3, 3))
    C[0, 0] = ct2 * ct1
    C[0, 1] = ct2 * st1
    C[0, 2] = -st2
    C[1, 0] = st3 * st2 * ct1 - ct3 * st1
    C[1, 1] = st3 * st2 * st1 + ct3 * ct1
    C[1, 2] = st3 * ct2
    C[2, 0] = ct3 * st2 * ct1 + st3 * st1
    C[2, 1] = ct3 * st2 * st1 - st3 * ct1
    C[2, 2] = ct3 * ct2

    return C


def C2Euler313(C):
    """
    Extract (3-1-3) Euler angles from Direction Cosine Matrix.

    Output: q = [alpha, beta, gamma] in radians
            alpha = first rotation about 3-axis (Z)
            beta  = second rotation about 1-axis (X)
            gamma = third rotation about 3-axis (Z)
    """
    q = np.zeros(3)
    q[0] = math.atan2(C[2, 0], -C[2, 1])  # alpha
    q[1] = math.acos(C[2, 2])              # beta
    q[2] = math.atan2(C[0, 2], C[1, 2])    # gamma

    return q


def Euler3132C(q):
    """
    Convert (3-1-3) Euler angles to Direction Cosine Matrix (for verification).
    """
    st1 = math.sin(q[0])
    ct1 = math.cos(q[0])
    st2 = math.sin(q[1])
    ct2 = math.cos(q[1])
    st3 = math.sin(q[2])
    ct3 = math.cos(q[2])

    C = np.zeros((3, 3))
    C[0, 0] = ct3 * ct1 - st3 * ct2 * st1
    C[0, 1] = ct3 * st1 + st3 * ct2 * ct1
    C[0, 2] = st3 * st2
    C[1, 0] = -st3 * ct1 - ct3 * ct2 * st1
    C[1, 1] = -st3 * st1 + ct3 * ct2 * ct1
    C[1, 2] = ct3 * st2
    C[2, 0] = st2 * st1
    C[2, 1] = -st2 * ct1
    C[2, 2] = ct2

    return C


def main():
    # Given (3-2-1) Euler angles in degrees
    psi = 10.0    # First rotation about 3-axis (Z)
    theta = 20.0  # Second rotation about 2-axis (Y)
    phi = 30.0    # Third rotation about 1-axis (X)

    print("=" * 60)
    print("Euler Angle Conversion: (3-2-1) to (3-1-3)")
    print("=" * 60)

    print("\nGiven (3-2-1) Euler angles:")
    print(f"  ψ (psi)   = {psi}°   (rotation about Z-axis)")
    print(f"  θ (theta) = {theta}°  (rotation about Y-axis)")
    print(f"  φ (phi)   = {phi}°  (rotation about X-axis)")

    # Convert to radians
    euler_321_rad = [math.radians(psi), math.radians(theta), math.radians(phi)]

    print(f"\n(3-2-1) Euler angles in radians:")
    print(f"  ψ = {euler_321_rad[0]:.10f} rad")
    print(f"  θ = {euler_321_rad[1]:.10f} rad")
    print(f"  φ = {euler_321_rad[2]:.10f} rad")

    # Step 1: Convert (3-2-1) Euler angles to Direction Cosine Matrix
    C = Euler3212C(euler_321_rad)

    print("\nStep 1: Direction Cosine Matrix [C] from (3-2-1) angles:")
    print(f"  [{C[0, 0]:12.8f}  {C[0, 1]:12.8f}  {C[0, 2]:12.8f}]")
    print(f"  [{C[1, 0]:12.8f}  {C[1, 1]:12.8f}  {C[1, 2]:12.8f}]")
    print(f"  [{C[2, 0]:12.8f}  {C[2, 1]:12.8f}  {C[2, 2]:12.8f}]")

    # Step 2: Extract (3-1-3) Euler angles from DCM
    euler_313_rad = C2Euler313(C)

    # Convert to degrees
    euler_313_deg = [math.degrees(euler_313_rad[0]),
                     math.degrees(euler_313_rad[1]),
                     math.degrees(euler_313_rad[2])]

    print("\n" + "=" * 60)
    print("RESULT: Equivalent (3-1-3) Euler angles")
    print("=" * 60)

    print(f"\nIn radians:")
    print(f"  α (alpha) = {euler_313_rad[0]:.10f} rad  (first rotation about Z)")
    print(f"  β (beta)  = {euler_313_rad[1]:.10f} rad  (second rotation about X)")
    print(f"  γ (gamma) = {euler_313_rad[2]:.10f} rad  (third rotation about Z)")

    print(f"\nIn degrees:")
    print(f"  α (alpha) = {euler_313_deg[0]:.6f}°")
    print(f"  β (beta)  = {euler_313_deg[1]:.6f}°")
    print(f"  γ (gamma) = {euler_313_deg[2]:.6f}°")

    # Verification
    print("\n" + "-" * 60)
    print("Verification: Converting (3-1-3) back to DCM...")

    C_verify = Euler3132C(euler_313_rad)

    print("\nDCM from (3-1-3) angles:")
    print(f"  [{C_verify[0, 0]:12.8f}  {C_verify[0, 1]:12.8f}  {C_verify[0, 2]:12.8f}]")
    print(f"  [{C_verify[1, 0]:12.8f}  {C_verify[1, 1]:12.8f}  {C_verify[1, 2]:12.8f}]")
    print(f"  [{C_verify[2, 0]:12.8f}  {C_verify[2, 1]:12.8f}  {C_verify[2, 2]:12.8f}]")

    # Check difference
    error = np.linalg.norm(C - C_verify)
    print(f"\nRotation matrix difference (Frobenius norm): {error:.2e}")

    if error < 1e-10:
        print("Verification successful - both angle sets produce the same rotation!")


if __name__ == "__main__":
    main()
