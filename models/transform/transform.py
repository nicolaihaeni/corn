import math
import numpy as np
import numpy.linalg as la

############################################
# Functions for common transformations
############################################

# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    identity = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(identity - shouldBeIdentity)
    return n < 1e-6


def quat2rot(q):
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    R = np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                  [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                  [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])
    if isRotm(R):
        return R
    else:
        raise Exception('R is not a rotation matrix, please check your quaternions')


# Convert rotation matrix to quaternion
def rot2quat(R):
    assert(isRotm(R))
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    # computing four sets of solutions
    qw_1 = math.sqrt(1 + r11 + r22 + r33)
    u1 = 1 / 2 * np.array([qw_1,
                          (r32 - r23) / qw_1,
                          (r13 - r31) / qw_1,
                          (r21 - r12) / qw_1])

    qx_2 = math.sqrt(1 + r11 - r22 - r33)
    u2 = 1 / 2 * np.array([(r32 - r23) / qx_2,
                          qx_2,
                          (r12 + r21) / qx_2,
                          (r31 + r13) / qx_2])

    qy_3 = math.sqrt(1 - r11 + r22 - r33)
    u3 = 1 / 2 * np.array([(r13 - r31) / qy_3,
                          (r12 + r21) / qy_3,
                          qy_3,
                          (r23 + r32) / qy_3])

    qz_4 = math.sqrt(1 - r11 - r22 + r33)
    u4 = 1 / 2 * np.array([(r21 - r12) / qz_4,
                          (r31 + r13) / qz_4,
                          (r32 + r23) / qz_4,
                          qz_4])

    U = [u1, u2, u3, u4]

    idx = np.array([r11 + r22 + r33, r11, r22, r33]).argmax()
    q = U[idx]
    if (la.norm(q) - 1) < 1e-3:
        return q
    else:
        raise Exception('Quaternion is not normalized, please check your rotation matrix')


def normalize_q(q):
    q /= la.norm(q)
    return q


def quat_product(q, r):
    t = np.zeros(4)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t


# Compute transformation between two rigid object poses
def compute_transformation(pose1, pose2, limits):
    p = pose1[:3] - pose2[:3]
    # Normalize position difference
    p[0] = p[0] / (limits[0][1] - limits[0][0])
    p[1] = p[1] / (limits[1][1] - limits[1][0])
    p[2] = p[2] / (limits[2][1] - limits[2][0])

    # Compute relative rotation
    q0 = pose1[3:]
    q0[0] = -q0[0]
    q0[1] = -q0[1]
    q0[2] = -q0[2]
    q1 = pose2[3:]

    q = quat_product(q1, q0)
    transformation = np.concatenate((p, q))
    return transformation
