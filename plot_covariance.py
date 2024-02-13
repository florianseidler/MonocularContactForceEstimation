import numpy as np
import open3d as o3d
from surface_mesh_methods.object_functions import untransformed_obj_mesh
from covariance_calculation import sharp_rotation_matrix
from pytransform3d.transformations._utils import check_transform
from pytransform3d.transformations._conversions import transform_from_exponential_coordinates
from pytransform3d.transformations._random import random_exponential_coordinates


object_name = '021_bleach_cleanser'  # '037_scissors'  021_bleach_cleanser

mean = np.array([[ 0.9769961 ,  0.20424134,  0.0610707 ,  0.22415799],
       [-0.20694056,  0.8387959 ,  0.50286438, -0.10619353],
       [ 0.05151883, -0.50393448,  0.86176655, -0.12955761],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

covariance = np.array([[ 0.05649426,  0.02731501,  0.02349848, -0.02363179, -0.01284085,
         0.00507521],
       [ 0.02731501,  0.2124997 , -0.22204871, -0.05948709, -0.03838643,
        -0.09547274],
       [ 0.02349848, -0.22204871,  0.71142413,  0.00599715, -0.03446634,
         0.07739432],
       [-0.02363179, -0.05948709,  0.00599715,  0.0330295 ,  0.02055649,
         0.03034569],
       [-0.01284085, -0.03838643, -0.03446634,  0.02055649,  0.02500677,
         0.02013905],
       [ 0.00507521, -0.09547274,  0.07739432,  0.03034569,  0.02013905,
         0.06583764]])


def random_transform(rng=np.random.default_rng(0), mean=np.eye(4), cov=np.eye(6)):
    r"""pytransform3d.transformations._random.random_transform() with strict_check==False"""
    mean = check_transform(mean, strict_check=False)
    Stheta = random_exponential_coordinates(rng=rng, cov=cov)
    delta = transform_from_exponential_coordinates(Stheta)
    return np.dot(delta, mean)


def get_transformed_object_mesh(seed, object_name):
    obj_mesh = untransformed_obj_mesh(object_name)
    trans_mat = random_transform(rng=np.random.default_rng(seed=seed), mean=mean, cov=covariance)
    trans_mat[0:3, 0:3] = sharp_rotation_matrix(matrix=trans_mat[0:3, 0:3], tolerance=1e-7)
    obj_mesh.transform(trans_mat)
    object_color = np.array([13, 114, 175, 255]) / 255.0  # blue
    obj_mesh.paint_uniform_color(object_color[:3])
    return obj_mesh


coord_sys_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
coord_sys_mesh_for_measure = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
trans_mat = np.eye(4)
trans_mat[:3,3] = np.array([0, 0, 0.1])
coord_sys_mesh_for_measure.transform(trans_mat)
coord_sys_mesh_for_measure_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
trans_mat = np.eye(4)
trans_mat[:3,3] = np.array([0, 0, -0.1])
coord_sys_mesh_for_measure_2.transform(trans_mat)
obj_mesh_untranslated = untransformed_obj_mesh(object_name)
combined_mesh = obj_mesh_untranslated + coord_sys_mesh + coord_sys_mesh_for_measure + coord_sys_mesh_for_measure_2

for i in range(20):
    obj_mesh = get_transformed_object_mesh(i, object_name)
    combined_mesh = combined_mesh + obj_mesh

o3d.visualization.draw_geometries([combined_mesh])
o3d.visualization.destroy_window()
