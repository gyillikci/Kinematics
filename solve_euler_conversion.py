#!/usr/bin/env python3
"""
Solve Euler Angle Conversion Problem
Convert (3-2-1) Euler angles to (3-1-3) Euler angles
"""

import numpy as np
import math
from RigidBodyKinematics import Euler3212C, C2Euler313, Euler3132C

# Given (3-2-1) Euler angles in degrees
theta1_deg = 20.0
theta2_deg = 10.0
theta3_deg = -10.0

print("=" * 60)
print("EULER ANGLE CONVERSION: (3-2-1) to (3-1-3)")
print("=" * 60)
print(f"\nGiven (3-2-1) Euler angles:")
print(f"  θ1 = {theta1_deg:7.3f}°")
print(f"  θ2 = {theta2_deg:7.3f}°")
print(f"  θ3 = {theta3_deg:7.3f}°")

# Convert to radians
theta1_rad = math.radians(theta1_deg)
theta2_rad = math.radians(theta2_deg)
theta3_rad = math.radians(theta3_deg)

# Create input vector for (3-2-1) Euler angles
q_321 = np.matrix([[theta1_rad], [theta2_rad], [theta3_rad]])

print(f"\nIn radians:")
print(f"  θ1 = {theta1_rad:10.6f} rad")
print(f"  θ2 = {theta2_rad:10.6f} rad")
print(f"  θ3 = {theta3_rad:10.6f} rad")

# Step 1: Convert (3-2-1) Euler angles to Direction Cosine Matrix
C = Euler3212C(q_321)

print(f"\nDirection Cosine Matrix (DCM):")
print(f"  [{C[0,0]:10.6f}, {C[0,1]:10.6f}, {C[0,2]:10.6f}]")
print(f"  [{C[1,0]:10.6f}, {C[1,1]:10.6f}, {C[1,2]:10.6f}]")
print(f"  [{C[2,0]:10.6f}, {C[2,1]:10.6f}, {C[2,2]:10.6f}]")

# Step 2: Convert Direction Cosine Matrix to (3-1-3) Euler angles
q_313 = C2Euler313(C)

# Convert to degrees
phi1_rad = q_313[0,0]
phi2_rad = q_313[1,0]
phi3_rad = q_313[2,0]

phi1_deg = math.degrees(phi1_rad)
phi2_deg = math.degrees(phi2_rad)
phi3_deg = math.degrees(phi3_rad)

print(f"\n" + "=" * 60)
print(f"RESULT: Equivalent (3-1-3) Euler angles:")
print(f"=" * 60)
print(f"  φ1 = {phi1_rad:10.6f} rad = {phi1_deg:9.4f}°")
print(f"  φ2 = {phi2_rad:10.6f} rad = {phi2_deg:9.4f}°")
print(f"  φ3 = {phi3_rad:10.6f} rad = {phi3_deg:9.4f}°")

print(f"\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

# Verify by converting back to DCM
C_verify = Euler3132C(q_313)
print(f"\nVerification DCM (should match original):")
print(f"  [{C_verify[0,0]:10.6f}, {C_verify[0,1]:10.6f}, {C_verify[0,2]:10.6f}]")
print(f"  [{C_verify[1,0]:10.6f}, {C_verify[1,1]:10.6f}, {C_verify[1,2]:10.6f}]")
print(f"  [{C_verify[2,0]:10.6f}, {C_verify[2,1]:10.6f}, {C_verify[2,2]:10.6f}]")

# Check if matrices are equal
diff = np.abs(C - C_verify)
max_error = np.max(diff)
print(f"\nMaximum element-wise error: {max_error:.2e}")

if max_error < 1e-10:
    print("✓ VERIFICATION PASSED: Matrices match!")
else:
    print("✗ VERIFICATION FAILED: Matrices do not match")

print(f"\n" + "=" * 60)
