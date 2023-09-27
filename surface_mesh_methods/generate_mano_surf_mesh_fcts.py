from paz import processors as pr
import pytransform3d.transformations as pt
import pytransform3d.rotations as pyrot
import numpy as np
from .mano import HandState
from surface_mesh_methods.hand_mesh.mano_pose_functions import predict_pose


def get_hand_state(kypt_3d_joints):

    world_xyz = [0, 0, 0]
    (mano_joint_angles, minimal_hand_absolute_joint_rotations,
        mano_joints_xyz, m0_translation, mpii_3d_joints) = predict_pose(kypt_3d_joints)
    wrap = pr.WrapOutput(
        ['world_xyz', 'mano_joints_xyz', 'mpii_3d_joints', 'mano_joint_angles',
        'minimal_hand_absolute_joint_rotations'])
    return wrap(world_xyz, mano_joints_xyz, mpii_3d_joints, mano_joint_angles,
                minimal_hand_absolute_joint_rotations), m0_translation


def get_transformation(transformation_values):
    """
    Get a Transformation Matrix from 6D coordinates.
    :param transformation_values: 6D coordinates
    :return: Transformation Matrix
    """
    transformation = pt.transform_from(pyrot.matrix_from_compact_axis_angle(
        [transformation_values[0], transformation_values[1], transformation_values[2]]),
        [transformation_values[3], transformation_values[4], transformation_values[5]])
    return transformation


def calc_mano_2_world(mano_2_world_transformation_values, hand_pose):
    """
    Convert Mano coordinates to world coordinates.
    :param mano_2_world_transformation_values
    :param hand_pose
    :return: converted coordinates
    """
    mano_matrix = get_transformation(mano_2_world_transformation_values)
    world_matrix = (pt.transform_from(pyrot.matrix_from_compact_axis_angle([0, 0, 0]), hand_pose['world_xyz']))
    # world_matrix = (pt.transform_from(pyrot.matrix_from_compact_axis_angle([0, 0, 0]), hand_pose['mano_joints_xyz']))
    mano_2_world = pt.concat(world_matrix, mano_matrix)
    return mano_2_world


def calc_mano_geometries(mano_shape_betas, hand_pose, mesh2world, left=False):
    """
    Get geometries of hand to apply them to mesh afterwards.
    :param mano_shape_betas:
    :param hand_pose:
    :param mesh2world:
    :param left:minimal_hand_handmesh_left
    :param usemediapipe:
    :return:
    """
    hand_state = HandState(left)
    hand_state.betas = mano_shape_betas
    joint_rotations = (flip_left_hand_joint_rotations_to_right_hand(hand_pose['mano_joint_angles']))
    hand_state.pose = np.ravel(joint_rotations[:16])
    hand_state.recompute_shape()
    hand_state.recompute_mesh(mesh2world=mesh2world)
    return hand_state


def print_hand_pose_values(hand_pose):
    print("world xyz: ", hand_pose["world_xyz"])
    print("mano joints xyz: ", hand_pose["mano_joints_xyz"])
    print("mpii 3d joints: ", hand_pose["mpii_3d_joints"])
    print("mano joint angles: ", hand_pose["mano_joint_angles"])
    print("minimal hand absolute joint rotations: ", hand_pose["minimal_hand_absolute_joint_rotations"])


def flip_left_hand_joint_rotations_to_right_hand(left_hand_joint_rotations):
    right_hand_joint_rotations = np.zeros(shape=(16, 3))
    for i in range(16):
        right_hand_joint_rotations[i] = left_hand_joint_rotations[i]
        right_hand_joint_rotations[i][2] = left_hand_joint_rotations[i][2] * -1
        right_hand_joint_rotations[i][1] = left_hand_joint_rotations[i][1] * -1
    return right_hand_joint_rotations