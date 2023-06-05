from paz import processors as pr
import pytransform3d.transformations as pt
import pytransform3d.rotations as pyrot
import numpy as np

from paz.applications import SSD512MinimalHandPose
from paz.backend.image import show_image, load_image
from paz.backend.camera import Camera
import argparse
import cv2

from paz.applications import MinimalHandPoseEstimation
from .mano import HandState

from .get_pose_ import predict_pose


def apply_kypt_estimator(input_image, seq_dir, frame):

    world_xyz = [0, 0, 0]
    (mano_joint_angles, minimal_hand_absolute_joint_rotations,
        mano_joints_xyz, mano_translation) = predict_pose(seq_dir, frame)
    #world_xyz = mano_translation
    # ab hier keine änderung
    mpii_3d_joints = None
    annotated_image = input_image
    wrap = pr.WrapOutput(
        ['world_xyz', 'mano_joints_xyz', 'mpii_3d_joints', 'mano_joint_angles',
        'minimal_hand_absolute_joint_rotations',
        'input_image', 'annotated_image'])
    return wrap(world_xyz, mano_joints_xyz, mpii_3d_joints, mano_joint_angles,
                minimal_hand_absolute_joint_rotations, input_image, annotated_image), mano_translation


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


def get_translation_from_2d_box(img_path):
    """
    Get the translation from the width of the 2d bounding box.
    :param img_path:
    :return: translation (x, y, z)
    """
    handpose_ssd512 = SSD512MinimalHandPose()
    image = load_image(img_path)
    detected_hand = handpose_ssd512.call(image)
    camera = Camera(img_path, cv2.CAP_IMAGES)
    # camera = Camera(args.camera_id)
    parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                        default=75, help='Horizontal field of view in degrees')
    args = parser.parse_args()
    camera.intrinsics_from_HFOV(args.horizontal_field_of_view)
    option = 1
    if option == 1:
        translation_box = pr.Translation3DFromBoxWidth(camera)
        translation = translation_box.call(detected_hand['boxes2D'])[0]
    else:
        solvePnP = pr.SolveChangingObjectPnPRANSAC(camera.intrinsics)
        detect = MinimalHandPoseEstimation(right_hand=True)(image)
        success, R, translation = solvePnP(detect['keypoints3D'], detect['keypoints2D'])
    print(translation)
    return translation


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