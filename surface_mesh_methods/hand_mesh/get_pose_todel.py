# currently unused

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import json
from keypoint_transformer.base import Tester
from keypoint_transformer.utils.vis import *
from keypoint_transformer.config import cfg 

from .kinematics import MANOHandJoints
from .hand_mesh import HandMesh
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr

from .iknet_network import iknet
from .kinematics import *  # mano_to_mpii, mpii_to_mano, xyz_to_delta, kypt_to_min_hand, MPIIHandJoints
import tensorflow as tf
from .paths import *


IK_UNIT_LENGTH = 0.09473151311686484 # in meter


def predict_pose(seq_dir, frame):
    """ Detects mano_params from images _via_kypt_trafo.

    # Arguments
        image: image shape (128,128)
    
    global_pos_joints: np.ndarray, shape [21, 3]  Joint locations XYZ.
    theta_mpii: np.ndarray, shape [21, 4]  Joint rotations Quat aus global_pos_joints.
    """
    global_pos_joints, theta_mpii, translation = process_via_kypt_trafo(seq_dir, frame) 
    absolute_angle_quaternions = mpii_to_mano(theta_mpii)  # quaternions
    joint_angles = calculate_handstate_joint_angles_from_min_hand_absolute_angles(absolute_angle_quaternions)

    global_pos_joints = mpii_to_mano(global_pos_joints)  # quaternions 

    return joint_angles, absolute_angle_quaternions, global_pos_joints, translation


def load_json(path):
  """
  Load pickle data.
  Parameter
  ---------
  path: Path to pickle file.
  Return
  ------
  Data in pickle file.
  """
  with open(path) as f:
    data = json.load(f)
    x = data
  return data


class ModelIK:
    """
  IKnet: estimating joint rotations from locations.
  """

    def __init__(self, input_size, network_fn, model_path, net_depth, net_width):
        """
    Parameters
    ----------
    input_size : int
      Number of joints to be used, e.g. 21, 42.
    network_fn : function
      Network function from `network.py`.
    model_path : str
      Path to the trained model.
    net_depth : int
      Number of layers.
    net_width : int
      Number of neurons in each layer.
    """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_ph = tf.compat.v1.placeholder(tf.float32, [1, input_size, 3])
            with tf.compat.v1.name_scope('network'):
                self.theta = \
                    network_fn(self.input_ph, net_depth, net_width, training=False)[0]
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.train.Saver().restore(self.sess, model_path)

    def process(self, joints):
        """
    Estimate joint rotations from locations.

    Parameters
    ----------
    joints : np.ndarray, shape [N, 3]
      Input joint locations (and other information e.g. bone orientation).

    Returns
    -------
    np.ndarray, shape [21, 4]
      Estimated global joint rotations in quaternions.
    """
        theta = \
            self.sess.run(self.theta, {self.input_ph: np.expand_dims(joints, 0)})
        if len(theta.shape) == 3:
            theta = theta[0]
        return theta
    

def process_via_kypt_trafo(seq_dir, frame):
    """
    Process a single frame.

    Returns
    -------
    np.ndarray, shape [21, 3]
    Joint locations.
    np.ndarray, shape [21, 4]
    Joint rotations.
    translation [3, 1]
    """

    joints_right, translation = get_right_mano_xyz(seq_dir, frame)
    joints_right = flip_joint_axis(joints_right, x=-1, y=-1, z=-1)
    xyz = kypt_to_min_hand(joints_right)  # kypt->mano->mpii, rel to M0, normed by |M0 - Wrist|

    delta, length = xyz_to_delta(xyz, MPIIHandJoints)
    delta *= length

    base_path = os.getcwd()
    ik_model = ModelIK(84, iknet, base_path + IK_MODEL_PATH, 6, 1024)
    
    left = True  # False
    if left:
        data = load_json(base_path + HAND_MESH_MODEL_LEFT_PATH_JSON)
    else:
        data = load_json(base_path + HAND_MESH_MODEL_RIGHT_PATH_JSON)

    mano_ref_xyz = data['joints']

    # convert the kinematic definition to MPII style, and normalize it
    mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
    mpii_ref_xyz -= mpii_ref_xyz[9:10]
    # get bone orientations in the reference pose
    mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
    mpii_ref_delta = mpii_ref_delta * mpii_ref_length

    pack = np.concatenate(
        [xyz, delta, mpii_ref_xyz, mpii_ref_delta], 0
    )

    theta = ik_model.process(pack)

    return xyz, theta, translation


def flip_joint_axis(joints, x=1, y=1, z=1):
  flipped_joints = np.zeros(shape=(21, 3))
  for i in range(21):
    flipped_joints[i] = joints[i]
    flipped_joints[i][0] = joints[i][0] * x
    flipped_joints[i][1] = joints[i][1] * y
    flipped_joints[i][2] = joints[i][2] * z
  return flipped_joints


def get_right_mano_xyz(seq_dir, frame):

    cfg.use_big_decoder = '--use_big_decoder'  # args.use_big_decoder
    cfg.dec_layers = 6  # args.dec_layers
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    # set gpu_ids
    cfg.set_args('0', '')

    tester = Tester('sources/snapshot_21_845.pth.tar')
    tester._make_batch_generator('test', 'all', None, None, None)
    #tester._make_single_frame_generator('test', 'all', None, None, None, seq_dir, frame)
    tester._make_model()

    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tester.batch_generator):
            # forward
            model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)
            out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}

    root_joint = torch.zeros((out['joint_3d_right'].shape[1], 1, 3)).to(out['joint_3d_right'].device)

    joints_right = out['joint_3d_right'][-1]
    joints_right = torch.cat([joints_right, root_joint], dim=1)  # .cpu().numpy()
    joints_right = joints_right.cpu().numpy()
    joints_right = joints_right[0]

    translation = out['rel_trans'].cpu().numpy()
    translation = translation[0]

    return joints_right, translation


def kypt_trafo():

    cfg.use_big_decoder = '--use_big_decoder'
    cfg.dec_layers = 6  # dec_layers
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    cfg.set_args('0', '')  # set gpu_ids

    tester = Tester('sources/snapshot_21_845.pth.tar')
    tester._make_batch_generator('test', 'all', None, None, None)
    tester._make_model()

    joints_right_vec = []
    translation_vec = []

    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tester.batch_generator):
            # forward
            model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)
            out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}

            root_joint = torch.zeros((out['joint_3d_right'].shape[1], 1, 3)).to(out['joint_3d_right'].device)

            joints_right = out['joint_3d_right'][-1]
            joints_right = torch.cat([joints_right, root_joint], dim=1)  # .cpu().numpy()
            joints_right = joints_right.cpu().numpy()
            joints_right_vec[itr] = joints_right[0]
            
            translation = out['rel_trans'].cpu().numpy()
            translation_vec[itr] = translation[0]

    return joints_right_vec, translation_vec


def calculate_handstate_joint_angles_from_min_hand_absolute_angles(quats):

    # rotate reference joints and get posed hand sceleton J
    J = rotated_ref_joints_from_quats(quats)

    # combine each joint with absolute rotation to transformation: t_posed_super_rotated
    t_posed_super_rotated = np.zeros(shape=(21, 4, 4))
    for i in range(21):
        t_posed_super_rotated[i] = pt.transform_from(
            pr.matrix_from_quaternion(quats[i]),
            J[i]
        )

    t_relative = np.zeros(shape=(21, 3, 3))

    # For each quaternion Q:
    for i in range(len(quats)):

        # Calc transformation with inverted rotation of Qi
        T_abs_rotations_i_inverted = pt.invert_transform(
                pt.transform_from(
                    pr.matrix_from_quaternion(quats[i]),
                    [0,0,0] #translation does not matter
                )
        )

        # Update Q_orientation if joint i has a parent (substract parents orientation)
        parent_index = MANOHandJoints.parents[i]
        if parent_index is not None:

            # Concatenate transformation get rotation difference (child to parent):
            # posed and super rotated joint i
            # inverted rotation of Qi
            t_posed_rotation_child_to_parent_i = pt.concat(
                t_posed_super_rotated[parent_index],
                T_abs_rotations_i_inverted
            )

            # clear out translationand keep only rotation
            t = pt.pq_from_transform(t_posed_rotation_child_to_parent_i)
            t_rotation_child_to_parent_i = np.array([t[3],t[4],t[5],t[6]])

            t_relative[i] = pr.matrix_from_quaternion(
                pr.q_conj(t_rotation_child_to_parent_i)
            )

    # Generate final array with 16 joint angles
    joint_angles = np.zeros(shape=(21, 3))

    # Root joint gets same orientation like absolute root quaternion
    joint_angles[0] = pr.compact_axis_angle_from_matrix(
        pr.matrix_from_quaternion(quats[0])
    )

    # Map of childs array_index = joint_index => parent_joint_index
    childs = [
        [1,4,7,10,13], # root_joint has multiple childs
        2,3,16,5,6,17,8,9,18,11,12,19,14,15,20 #other joints have exactly one parent
    ]
    # Joint 1-15 gets calculated orientation of child's join
    for i in range(1,16):
        joint_angles[i] = pr.compact_axis_angle_from_matrix(
            t_relative[childs[i]]
        )
    
    #joint_angles = np.ravel(joint_angles[:16])

    return joint_angles

"""
Rotate reference joints by estimated absolute quats
"""
def rotated_ref_joints_from_quats(quat):
    rotation_matrices = np.zeros(shape=(21, 3, 3))
    for j in range(len(quat)):
        rotation_matrices[j] = pr.matrix_from_quaternion(quat[j])
    mats = np.stack(rotation_matrices, 0)
    hand_mesh = HandMesh(left=True) #Attetion, decides which ref_pose is loaded! should be left here
    joint_xyz = np.matmul(mats, hand_mesh.ref_pose)[..., 0]
    return joint_xyz
