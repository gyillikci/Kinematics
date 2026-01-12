#!/usr/bin/env python3
"""
Relative Attitude Calculation

Given:
- B relative to N: (3-2-1) Euler angles (10, 20, 30) degrees
- R relative to N: (3-2-1) Euler angles (-5, 5, 5) degrees

Find: B relative to R in (3-2-1) Euler angles
"""

import numpy as np
import math


def Euler3212C(q):
    """
    Convert (3-2-1) Euler angles to Direction Cosine Matrix.
    Input: q = [psi, theta, phi] in radians
    """
    st1 = math.sin(q[0])
    ct1 = math.cos(q[0])
    st2 = math.sin(q[1])
    ct2 = math.cos(q[1])
    st3 = math.sin(q[2])
    ct3 = math.cos(q[2])

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


def C2Euler321(C):
    """
    Extract (3-2-1) Euler angles from Direction Cosine Matrix.
    Output: q = [psi, theta, phi] in radians
    """
    q = np.zeros(3)
    q[0] = math.atan2(C[0, 1], C[0, 0])  # psi
    q[1] = math.asin(-C[0, 2])            # theta
    q[2] = math.atan2(C[1, 2], C[2, 2])   # phi

    return q


def main():
    # Given (3-2-1) Euler angles in degrees
    # B relative to N
    BN_deg = [10.0, 20.0, 30.0]
    # R relative to N
    RN_deg = [-5.0, 5.0, 5.0]

    print("=" * 60)
    print("Relative Attitude: B relative to R")
    print("=" * 60)

    print("\nGiven:")
    print(f"  B relative to N (3-2-1): ({BN_deg[0]}°, {BN_deg[1]}°, {BN_deg[2]}°)")
    print(f"  R relative to N (3-2-1): ({RN_deg[0]}°, {RN_deg[1]}°, {RN_deg[2]}°)")

    # Convert to radians
    BN_rad = [math.radians(a) for a in BN_deg]
    RN_rad = [math.radians(a) for a in RN_deg]

    # Step 1: Compute DCM [BN] - B relative to N
    C_BN = Euler3212C(BN_rad)
    print("\nDCM [BN] (B relative to N):")
    print(f"  [{C_BN[0, 0]:12.8f}  {C_BN[0, 1]:12.8f}  {C_BN[0, 2]:12.8f}]")
    print(f"  [{C_BN[1, 0]:12.8f}  {C_BN[1, 1]:12.8f}  {C_BN[1, 2]:12.8f}]")
    print(f"  [{C_BN[2, 0]:12.8f}  {C_BN[2, 1]:12.8f}  {C_BN[2, 2]:12.8f}]")

    # Step 2: Compute DCM [RN] - R relative to N
    C_RN = Euler3212C(RN_rad)
    print("\nDCM [RN] (R relative to N):")
    print(f"  [{C_RN[0, 0]:12.8f}  {C_RN[0, 1]:12.8f}  {C_RN[0, 2]:12.8f}]")
    print(f"  [{C_RN[1, 0]:12.8f}  {C_RN[1, 1]:12.8f}  {C_RN[1, 2]:12.8f}]")
    print(f"  [{C_RN[2, 0]:12.8f}  {C_RN[2, 1]:12.8f}  {C_RN[2, 2]:12.8f}]")

    # Step 3: Compute [BR] = [BN][NR] = [BN][RN]^T
    # [NR] = [RN]^T (transpose gives inverse for orthogonal matrices)
    C_NR = C_RN.T
    C_BR = C_BN @ C_NR

    print("\nDCM [BR] = [BN][RN]^T (B relative to R):")
    print(f"  [{C_BR[0, 0]:12.8f}  {C_BR[0, 1]:12.8f}  {C_BR[0, 2]:12.8f}]")
    print(f"  [{C_BR[1, 0]:12.8f}  {C_BR[1, 1]:12.8f}  {C_BR[1, 2]:12.8f}]")
    print(f"  [{C_BR[2, 0]:12.8f}  {C_BR[2, 1]:12.8f}  {C_BR[2, 2]:12.8f}]")

    # Step 4: Extract (3-2-1) Euler angles from [BR]
    BR_rad = C2Euler321(C_BR)
    BR_deg = [math.degrees(a) for a in BR_rad]

    print("\n" + "=" * 60)
    print("RESULT: B relative to R in (3-2-1) Euler angles")
    print("=" * 60)

    print(f"\nIn radians:")
    print(f"  ψ (psi)   = {BR_rad[0]:.10f} rad")
    print(f"  θ (theta) = {BR_rad[1]:.10f} rad")
    print(f"  φ (phi)   = {BR_rad[2]:.10f} rad")

    print(f"\nIn degrees:")
    print(f"  ψ (psi)   = {BR_deg[0]:.6f}°")
    print(f"  θ (theta) = {BR_deg[1]:.6f}°")
    print(f"  φ (phi)   = {BR_deg[2]:.6f}°")

    # Verification
    print("\n" + "-" * 60)
    print("Verification: [BR][RN] should equal [BN]")
    C_BN_verify = C_BR @ C_RN
    error = np.linalg.norm(C_BN - C_BN_verify)
    print(f"Error (Frobenius norm): {error:.2e}")
    if error < 1e-10:
        print("Verification successful!")


if __name__ == "__main__":
    main()
