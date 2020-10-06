import numpy as np
import torch

def get_object_T_camera(x: float, y: float, z: float, get_4x4: bool = False) -> np.ndarray:
    z_vector = np.array([-x, -y, -z])
    e_z = z_vector / np.linalg.norm(z_vector)
    x_vector = np.cross(e_z, np.array([0, 0, 1]))
    e_x = x_vector / np.linalg.norm(x_vector)
    e_y = np.cross(e_z, e_x)

    camera_position = np.array([x, y, z])

    object_T_camera = np.c_[e_x, e_y, e_z, camera_position]

    if get_4x4:
        object_T_camera = np.vstack((object_T_camera, np.array([0., 0., 0., 1.])))

    return object_T_camera


def spherical_to_cartesian(azimuth: float, elevation: float, distance: float = 1.0):
    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)

    return x, y, z


def pose_from_filename(filename: str, get_4x4: bool = True) -> np.ndarray:
    azimuth_degree, elevation_degree = tuple(float(v) for v in filename.split('.')[0].split('_')[-2:])
    azimuth_degree *= -10

    azimuth_rad, elevation_rad = np.deg2rad(azimuth_degree), np.deg2rad(elevation_degree)
    x, y, z = spherical_to_cartesian(azimuth_rad, elevation_rad)

    object_T_camera = get_object_T_camera(x, y, z, get_4x4)

    return object_T_camera


def rotmat_to_quat(R: np.ndarray, get_wxyz: bool = False) -> np.ndarray:
    m00 = R[0,0];    m01 = R[0,1];    m02 = R[0,2];    
    m10 = R[1,0];    m11 = R[1,1];    m12 = R[1,2];    
    m20 = R[2,0];    m21 = R[2,1];    m22 = R[2,2]

    qw= np.sqrt(1 + m00 + m11 + m22) /2
    if qw == 0:
        qw = 0.0001
    qx = (m21 - m12)/(4*qw)
    qy = (m02 - m20)/(4*qw)
    qz = (m10 - m01)/(4*qw)

    if get_wxyz:
        q = np.array([qw, qx, qy, qz])
    else:
        q = np.array([qx, qy, qz, qw])
    q = q/np.linalg.norm(q)

    return q
