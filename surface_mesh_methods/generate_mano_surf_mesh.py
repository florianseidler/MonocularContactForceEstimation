import pytransform3d.transformations as pt
import numpy as np

from .Transformation_Helper import TransformationHelper
from .O3d_vis_utils import O3dVisUtils
from .generate_mano_surf_mesh_fcts import (calc_mano_2_world, calc_mano_geometries, get_hand_state)
from .mano_beta_shape_estimation.beta_shape_estimation import pso_betas


def get_kypt_mesh(kypt_3d_joints, particle_swarm_optimization=1):

    hand_pose, m0_translation = get_hand_state(kypt_3d_joints) # without shape params

    # Calculate mano 2 world to transform coordinates
    mano_2_world_transformation_values = [0, 0, 0, 0, 0, 0]
    mano_2_world = calc_mano_2_world(mano_2_world_transformation_values, hand_pose)

    # Adjust Mano Beta Shape params
    if particle_swarm_optimization:
        mano_shape_betas = pso_betas(hand_pose["mpii_3d_joints"]).reshape(-1)  # Particle Swarm Optimization
    else:
        mano_shape_betas = np.zeros(10)
    
    # Calculate render meshes
    mesh2world = pt.concat(TransformationHelper().mano_2_world_base(
        mano_shape_betas=mano_shape_betas, left=False), mano_2_world)

    # Calculate MANO Right Geometries
    hand_state = calc_mano_geometries(mano_shape_betas, hand_pose, mesh2world, left=False)

    # Generate Hand Mesh
    mano_mesh = O3dVisUtils.make_mano_mesh(hand_state.vertices, hand_state.faces,
                                           handVertContact=None, handVertIntersec=None)

    return mano_mesh
