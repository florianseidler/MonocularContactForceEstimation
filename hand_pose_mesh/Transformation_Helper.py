import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr

from .mano import HandState
from .hand_mesh import HandMesh


class TransformationHelper:

    def __init__(self):
        self.hand_state_left = HandState(left=True)
        self.hand_state_right = HandState(left=False)
        self.minimal_hand_handmesh_left = HandMesh(left=True)  # SUPPORTS ONLY LEFT?
        self.minimal_hand_handmesh_right = HandMesh(left=False)  # SUPPORTS ONLY LEFT?
        print("TransformationHelper initialized")

    """
    Returns absolute MANO joint values
    """

    def get_mano_base_joint(self, mano_shape_betas=None, left=True):
        return self.get_mano_base_joints(mano_shape_betas=mano_shape_betas, left=left)[0]

    def get_mano_base_joints(self, mano_shape_betas=None, left=True):
        if left:
            hand_state = self.hand_state_left
        else:
            hand_state = self.hand_state_right
        zero_rotations = np.zeros(shape=(16, 3))
        hand_state.pose = np.ravel(zero_rotations)
        hand_state.betas = mano_shape_betas
        hand_state.recompute_shape()
        hand_state.recompute_mesh(mesh2world=None)
        joints_xyz = hand_state.get_current_joints()
        return joints_xyz

    def get_minimalhand_base_joint(self, minimal_hand_absolute_rotations=np.zeros(shape=(21, 4)), left=True):
        return \
        self.get_minimalhand_base_joints(minimal_hand_absolute_rotations=minimal_hand_absolute_rotations, left=left)[0]

    def get_minimalhand_base_joints(self, minimal_hand_absolute_rotations=np.zeros(shape=(21, 4)), left=True):
        if left:
            min_handmesh = self.minimal_hand_handmesh_left
        else:
            min_handmesh = self.minimal_hand_handmesh_right
        min_hand_joints_xyz = min_handmesh.get_joint_xyz_by_abs_quats(minimal_hand_absolute_rotations)[:16]
        return min_hand_joints_xyz

    def robot_2_world_base(self):
        return self.mano_2_world_base(mano_shape_betas=np.zeros(10), left=True)  # TODO change to robot base

    def robot_2_mano(self, robot_2_mano_transformation_setting=None, mano_shape_betas=None,
                     mano_joint_angles=np.zeros(shape=(16, 3)),
                     left=True):
        """
        # calc robot_2_world
        if self.visibility_settings['robot_mesh_shadow_hand']['selected'] or \
                self.visibility_settings['robot_mesh_mia_hand']['selected']:
            mano_right_root_rotation_transform = pt.transform_from(
                pr.matrix_from_compact_axis_angle(mano_rotations_right[0]),
                default_mano_right_joints[0])

            mano_right_root_translation_transform = pt.transform_from(
                pr.matrix_from_compact_axis_angle([0, 0, 0]),
                default_mano_right_joints[0] * -1)

            robot_2_mano_with_orientation_transform = pt.concat(
                mano_right_root_translation_transform,
                mano_right_root_rotation_transform
            )
            robot_2_world = pt.concat(
                self.transformation_helper.robot_2_mano(),
                mano_2_world
            )
        """
        mano_base_joint = self.get_mano_base_joint(mano_shape_betas=mano_shape_betas, left=left)
        mano_wrist_orientation = mano_joint_angles[0]
        return pt.concat(
            pt.transform_from(
                pr.matrix_from_compact_axis_angle(mano_wrist_orientation),
                mano_base_joint),
            robot_2_mano_transformation_setting
        )

    def mano_2_world_base(self, mano_shape_betas=None, left=True, factor_plus=1):
        mano_2_base = self.get_mano_base_joint(mano_shape_betas=mano_shape_betas, left=left)
        factor = -1

        factor = factor * factor_plus
        t = pt.transform_from(TransformationHelper.zero_rotations(),
                                 [mano_2_base[0] * factor,
                                  mano_2_base[1] * factor,
                                  mano_2_base[2] * factor])
        return t

    def minimalhand_2_world_base(self, left=True):
        minimalhand_2_base = self.get_minimalhand_base_joint(left=left)
        return pt.transform_from(TransformationHelper.zero_rotations(),
                                 [minimalhand_2_base[0] * -1,
                                  minimalhand_2_base[1] * -1,
                                  minimalhand_2_base[2] * -1])

    """
      Generate Transformation MinimalHand_Base -> MANO_Handstate_Base

      joint_rotations (16,3) 
          joint rotations in axis angle representation (MANO handstate rotations)
      min_hand_quats (21,4)
          rotations quaterions from minimal_hand_estimator output
      """

    def minimal_hand_2_mano_transformation(self, minimal_hand_absolute_rotations=np.zeros(shape=(21, 4)),
                                           mano_left=True, minimalhand_left=True):
        mano_base_joint = self.get_mano_base_joint(mano_shape_betas=np.zeros(10), left=mano_left)
        minimalhand_base_joint = self.get_minimalhand_base_joint(
            minimal_hand_absolute_rotations=minimal_hand_absolute_rotations,
            left=minimalhand_left
        )
        return pt.transform_from(TransformationHelper.zero_rotations(),
                                 [mano_base_joint[0] - minimalhand_base_joint[0],
                                  mano_base_joint[1] - minimalhand_base_joint[1],
                                  mano_base_joint[2] - minimalhand_base_joint[2]])

    def get_world_2_robot_transformation(self, mesh_2_world=None, mano_wrist_joint_angles=np.zeros(3),
                                         mano_shape_betas=np.zeros(10), mano_left=False):
        robot_2_default_mano_right_with_orientation = pt.concat(
            self.mano_2_world_base(mano_shape_betas=mano_shape_betas, left=mano_left),
            TransformationHelper.get_mano_orientation_transformation(mano_wrist_joint_angles=mano_wrist_joint_angles),
        )
        return pt.concat(
            robot_2_default_mano_right_with_orientation,
            mesh_2_world
        )

    @staticmethod
    def get_mano_orientation_transformation(mano_wrist_joint_angles=np.zeros(3)):
        mano_orientation = pt.transform_from(
            pr.matrix_from_compact_axis_angle(mano_wrist_joint_angles),
            [0, 0, 0]
        )
        return mano_orientation

    def get_object_2_world(self,
                           mano_2_world_transform=None,
                           mano_world_xyz=np.zeros(3),
                           mano_wrist_joint_angles=np.zeros(3),
                           mano_shape=np.zeros(10),
                           objRot=np.zeros(3),
                           objTrans=np.zeros(3),
                           wrist_zero_translation=False,
                           wrist_zero_rotation=False
                           ):
        t = self.zero_transformation()


        t = pt.concat(
            self.zero_transformation(),
            pt.transform_from(
                pr.matrix_from_compact_axis_angle(objRot),
                [0, 0, 0]
            ),
        )
        t = pt.concat(
            t,
            pt.transform_from(
                self.zero_rotations(),
                objTrans
            ),
        )

        t = pt.concat(
            t,
            self.mano_2_world_base(mano_shape_betas=mano_shape, left=False),

        )
        # reset object translation
        if wrist_zero_translation:
            t = pt.concat(
                t,
                pt.invert_transform(
                    pt.transform_from(
                        pr.matrix_from_compact_axis_angle([0, 0, 0]),
                        mano_world_xyz
                    )
                ),
            )
        if wrist_zero_rotation:
            t = pt.concat(
                t,
                pt.invert_transform(
                    self.get_mano_orientation_transformation(
                        mano_wrist_joint_angles=mano_wrist_joint_angles)
                ),
            )
        t = pt.concat(
            t,
            mano_2_world_transform
        )
        return t

    def get_robot_2_world_transformation(self, mesh_2_world=None, robot_2_mano_transformation_setting=None,
                                         mano_shape_betas=np.zeros(10), mano_wrist_joint_angles=np.zeros(3)):

        robot_2_default_mano_right_with_orientation = pt.concat(
            self.mano_2_world_base(mano_shape_betas=mano_shape_betas, left=False),
            TransformationHelper.get_mano_orientation_transformation(mano_wrist_joint_angles=mano_wrist_joint_angles),
            # here better mano joint_angles 0? instead of mapped ones!
        )
        return pt.concat(
            pt.concat(
                pt.concat(
                    pt.invert_transform(robot_2_mano_transformation_setting),
                    robot_2_default_mano_right_with_orientation,
                ),
                pt.concat(
                    mesh_2_world,
                    pt.invert_transform(
                        self.mano_2_world_base(mano_shape_betas=mano_shape_betas,
                                               left=False))),
            ),
            self.mano_2_world_base(mano_shape_betas=mano_shape_betas, left=False)
        )

    @staticmethod
    def app_view_transformation():
        r = pr.matrix_from_compact_axis_angle([3.2, 0, 0])  # Rotation Matrix
        t = pt.transform_from(r, [0.15, 0.05, 0.05])  # Translation
        return t

    @staticmethod
    def zero_transformation():
        # Transform and Rotate to World
        r = pr.matrix_from_compact_axis_angle([0, 0, 0])  # Rotation Matrix
        t = pt.transform_from(r, [0, 0, 0])  # Translation
        return t

    @staticmethod
    def zero_rotations():
        return pr.matrix_from_compact_axis_angle([0, 0, 0])  # Rotation Matrix

    @staticmethod
    def invert_x_rotation():
        return pr.matrix_from_compact_axis_angle([-1, 0, 0])  # Rotation Matrix

    @staticmethod
    def move_right_transformation():
        r = pr.matrix_from_compact_axis_angle([0, 0, 0])  # Rotation Matrix
        t = pt.transform_from(r, [0.2, 0, 0])  # Translation
        return t
