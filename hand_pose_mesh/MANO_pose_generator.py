import cv2
import pytransform3d.transformations as pt
import numpy as np

from .Transformation_Helper import TransformationHelper
from .O3d_vis_utils import O3dVisUtils
from .MANO_pose_generator_functions import (calc_mano_2_world, calc_mano_geometries,
                                            apply_kypt_estimator)


def generate_mano_mesh_from_kypt(img_path, seq_dir, frame):
    # Load image
    image = cv2.imread(img_path)

    hand_pose, mano_translation = apply_kypt_estimator(image, seq_dir, frame)

    # Calculate mano 2 world to transform coordinates
    mano_2_world_transformation_values = [0, 0, 0, 0, 0, 0]
    mano_2_world = calc_mano_2_world(mano_2_world_transformation_values, hand_pose)

    # Calculate render meshes
    mano_shape_betas = np.zeros(10)
    mesh2world = pt.concat(TransformationHelper().mano_2_world_base(
        mano_shape_betas=mano_shape_betas, left=False), mano_2_world)

    # Calculate MANO Right Geometries
    hand_state = calc_mano_geometries(mano_shape_betas, hand_pose, mesh2world, left=False)

    # Generate Hand Mesh
    mano_mesh = O3dVisUtils.make_mano_mesh(hand_state.vertices, hand_state.faces,
                                           handVertContact=None, handVertIntersec=None)

    return mano_mesh  # , hand_state
