"""
vector3D.py

Python conversion of vector3D.c/h library
Created by Hanspeter Schaub on Sat Mar 01 2003.
Converted to Python on 2025-11-10

Provides a library for doing 3D matrix algebra using NumPy.
"""

import numpy as np
import math

# Constants
M_PI = math.pi
D2R = M_PI / 180.0
R2D = 180.0 / M_PI


def set3(x, y, z):
    """Create a 3D vector from three scalar values."""
    return np.array([x, y, z], dtype=float)


def setMatrix(m11, m12, m13, m21, m22, m23, m31, m32, m33):
    """Create a 3x3 matrix from nine scalar values."""
    return np.array([
        [m11, m12, m13],
        [m21, m22, m23],
        [m31, m32, m33]
    ], dtype=float)


def set4(x, y, z, w):
    """Create a 4D vector from four scalar values."""
    return np.array([x, y, z, w], dtype=float)


def setZero():
    """Create a zero 3D vector."""
    return np.zeros(3, dtype=float)


def norm(x):
    """Calculate the norm (magnitude) of a 3D vector."""
    return np.linalg.norm(x)


def norm4(x):
    """Calculate the norm (magnitude) of a 4D vector."""
    return np.linalg.norm(x)


def cross(x, y):
    """Calculate the cross product of two 3D vectors."""
    return np.cross(x, y)


def mult(g, x):
    """Multiply a vector by a scalar."""
    return g * x


def equal(y):
    """Create a copy of a vector."""
    return y.copy()


def dot(x, y):
    """Calculate the dot product of two vectors."""
    return np.dot(x, y)


def add(x, y):
    """Add two vectors."""
    return x + y


def sub(x, y):
    """Subtract two vectors."""
    return x - y


def printVector(str_label, vec):
    """Print a 3D vector with a label."""
    print(f"{str_label} ({vec[0]:20.15g}, {vec[1]:20.15g}, {vec[2]:20.15g})")


def printVector4(str_label, vec):
    """Print a 4D vector with a label."""
    print(f"{str_label} ({vec[0]:20.15g}, {vec[1]:20.15g}, {vec[2]:20.15g}, {vec[3]:20.15g})")


def printMatrix(str_label, mat):
    """Print a 3x3 matrix with a label."""
    print(f"{str_label}:")
    print(f"{mat[0][0]:20.15g}, {mat[0][1]:20.15g}, {mat[0][2]:20.15g}")
    print(f"{mat[1][0]:20.15g}, {mat[1][1]:20.15g}, {mat[1][2]:20.15g}")
    print(f"{mat[2][0]:20.15g}, {mat[2][1]:20.15g}, {mat[2][2]:20.15g}")


def equalCheck(v1, v2, d):
    """
    Check if two 3D vectors are equal within a tolerance.
    Returns True if all elements differ by less than 10^(-d).
    """
    tolerance = 10.0 ** (-d)
    return np.all(np.abs(v1 - v2) <= tolerance)


def MequalCheck(m1, m2, d):
    """
    Check if two 3x3 matrices are equal within a tolerance.
    Returns True if all elements differ by less than 10^(-d).
    """
    tolerance = 10.0 ** (-d)
    return np.all(np.abs(m1 - m2) <= tolerance)


def VequalCheck(a, b, d):
    """
    Check if two 3D vectors are equal within a tolerance.
    Returns True if all elements differ by less than 10^(-d).
    """
    return equalCheck(a, b, d)


def V4equalCheck(a, b, d):
    """
    Check if two 4D vectors are equal within a tolerance.
    Returns True if all elements differ by less than 10^(-d).
    """
    tolerance = 10.0 ** (-d)
    return np.all(np.abs(a - b) <= tolerance)


def dotT(v1, v2):
    """
    Calculate the outer product of two 3D vectors (v1 * v2^T).
    Returns a 3x3 matrix.
    """
    return np.outer(v1, v2)


def Mdot(m, a):
    """Multiply a 3x3 matrix by a 3D vector."""
    return np.dot(m, a)


def MdotM(m1, m2):
    """Multiply two 3x3 matrices."""
    return np.dot(m1, m2)


def MdotMT(m1, m2):
    """Multiply a 3x3 matrix by the transpose of another."""
    return np.dot(m1, m2.T)


def transpose(m):
    """Transpose a 3x3 matrix."""
    return m.T


def Mmult(a, m):
    """Multiply a 3x3 matrix by a scalar."""
    return a * m


def Madd(m1, m2):
    """Add two 3x3 matrices."""
    return m1 + m2


def Msub(m1, m2):
    """Subtract two 3x3 matrices."""
    return m1 - m2


def tilde(a):
    """
    Create the skew-symmetric (tilde) matrix from a 3D vector.
    Used for cross product representation: tilde(a) @ b = cross(a, b)
    """
    return np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0]
    ], dtype=float)


def inverse(m):
    """
    Calculate the inverse of a 3x3 matrix.
    Returns NaN matrix if the matrix is singular.
    """
    det = detM(m)
    if abs(det) > 1e-12:
        return np.linalg.inv(m)
    else:
        import warnings
        warnings.warn("ERROR: singular 3x3 matrix inverse")
        return np.full((3, 3), np.nan)


def detM(m):
    """Calculate the determinant of a 3x3 matrix."""
    return np.linalg.det(m)


def trace(m):
    """Calculate the trace of a 3x3 matrix."""
    return np.trace(m)


def eye():
    """Create a 3x3 identity matrix."""
    return np.eye(3, dtype=float)


def arc_cosh(z):
    """Calculate the inverse hyperbolic cosine (arcosh)."""
    return math.log(z + math.sqrt(z + 1.0) * math.sqrt(z - 1.0))


def arc_tanh(z):
    """Calculate the inverse hyperbolic tangent (arctanh)."""
    return 0.5 * (math.log(1.0 + z) - math.log(1.0 - z))
