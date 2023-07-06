import numpy as np


class MANOHandJoints:
  n_joints = 21

  labels = [
    'W', #0
    'I0', 'I1', 'I2', #3
    'M0', 'M1', 'M2', #6
    'L0', 'L1', 'L2', #9
    'R0', 'R1', 'R2', #12
    'T0', 'T1', 'T2', #15
    'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
  ]

  # finger tips are not joints in MANO, we label them on the mesh manually
  mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

  parents = [
    None,
    0, 1, 2,
    0, 4, 5,
    0, 7, 8,
    0, 10, 11,
    0, 13, 14,
    3, 6, 9, 12, 15
  ]


class MPIIHandJoints:
  n_joints = 21

  labels = [
    'W', #0
    'T0', 'T1', 'T2', 'T3', #4
    'I0', 'I1', 'I2', 'I3', #8
    'M0', 'M1', 'M2', 'M3', #12
    'R0', 'R1', 'R2', 'R3', #16
    'L0', 'L1', 'L2', 'L3', #20
  ]

  parents = [
    None,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19
  ]


class SkeletonJoints:
  n_joints = 21
  '''
  labels = [
    'T4', 'T3', 'T2', 'T1', #3
    'I4', 'I3', 'I2', 'I1', #7
    'M4', 'M3', 'M2', 'M1', #11
    'R4', 'R3', 'R2', 'R1', #15
    'L4', 'L3', 'L2', 'L1', #19
    'W' #20
  ]
  
  labels = [  # bei den *3 anpassen
    'T3', 'T2', 'T1', 'T0', #3
    'I3', 'I2', 'I1', 'I0', #7
    'M3', 'M2', 'M1', 'M0', #11
    'R3', 'R2', 'R1', 'R0', #15
    'L3', 'L2', 'L1', 'L0', #19
    'W' #20
  ]
  '''
  labels = [
    'I3', 'T2', 'T1', 'T0', #3
    'M3', 'I2', 'I1', 'I0', #7
    'L3', 'M2', 'M1', 'M0', #11
    'R3', 'R2', 'R1', 'R0', #15
    'T3', 'L2', 'L1', 'L0', #19
    'W' #20
  ]

  parents = [
    1, 2, 3, 20,
    5, 6, 7, 20,
    9, 10, 11, 20,
    13, 14, 15, 20,
    17, 18, 19, 20,
    None
  ]


class SkeletonJoints_real:
  n_joints = 21
  
  labels = [  # bei den *3 anpassen
    'T3', 'T2', 'T1', 'T0', #3
    'I3', 'I2', 'I1', 'I0', #7
    'M3', 'M2', 'M1', 'M0', #11
    'R3', 'R2', 'R1', 'R0', #15
    'L3', 'L2', 'L1', 'L0', #19
    'W' #20
  ]


  parents = [
    1, 2, 3, 20,
    5, 6, 7, 20,
    9, 10, 11, 20,
    13, 14, 15, 20,
    17, 18, 19, 20,
    None
  ]


class SkeletonJointsMinHand:
  n_joints = 21
  labels = [
    'I3', 'T2', 'T1', 'T0', #3   # T
    'M3', 'R2', 'M1', 'I0', #7   # I
    'L3', 'M2', 'R1', 'M0', #11  # M
    'R3', 'L2', 'L1', 'R0', #15  # R
    'T3', 'I2', 'I1', 'L0', #19  # L
    'W' #20
  ]

  parents = [
    1, 2, 3, 20,
    5, 6, 7, 20,
    9, 10, 11, 20,
    13, 14, 15, 20,
    17, 18, 19, 20,
    None
  ]


def mpii_to_mano(mpii):
  """
  Map data from MPIIHandJoints order to MANOHandJoints order.

  Parameters
  ----------
  mpii : np.ndarray, [21, ...]
    Data in MPIIHandJoints order. Note that the joints are along axis 0.

  Returns
  -------
  np.ndarray
    Data in MANOHandJoints order.
  """
  mano = []
  for j in range(MANOHandJoints.n_joints):
    mano.append(
      mpii[MPIIHandJoints.labels.index(MANOHandJoints.labels[j])]
    )
  mano = np.stack(mano, 0)
  return mano


def mano_to_mpii(mano):
  """
  Map data from MANOHandJoints order to MPIIHandJoints order.

  Parameters
  ----------
  mano : np.ndarray, [21, ...]
    Data in MANOHandJoints order. Note that the joints are along axis 0.

  Returns
  -------
  np.ndarray
    Data in MPIIHandJoints order.
  """
  mpii = []
  for j in range(len(mano)):
    mpii.append(
      mano[MANOHandJoints.labels.index(MPIIHandJoints.labels[j])]
    )
  mpii = np.stack(mpii, 0)
  return mpii


def skeleton_to_mano(skeleton):
  """
  Map data from SkeletonJoints order to MANOHandJoints order.

  Parameters
  ----------
  skeleton : np.ndarray, [21, ...]
    Data in SkeletonJoints order. Note that the joints are along axis 0.

  Returns
  -------
  np.ndarray
    Data in MANOHandJoints order.
  """
  mano = []
  for j in range(MANOHandJoints.n_joints):
    mano.append(skeleton[SkeletonJoints.labels.index(MANOHandJoints.labels[j])])
  mano = np.stack(mano, 0)
  return mano


def skeleton_to_mano_real(skeleton):
  mano = []
  for j in range(MANOHandJoints.n_joints):
    mano.append(skeleton[SkeletonJoints_real.labels.index(MANOHandJoints.labels[j])])
  mano = np.stack(mano, 0)
  return mano


def mano_to_skeleton(mano):
  """
  Map data from MANOHandJoints order to SkeletonJoints order.

  Parameters
  ----------
  mano : np.ndarray, [21, ...]
    Data in MANOHandJoints order. Note that the joints are along axis 0.

  Returns
  -------
  np.ndarray
    Data in SkeletonJoints order.
  """
  skeleton = []
  for j in range(len(mano)):
    skeleton.append(
      mano[MANOHandJoints.labels.index(SkeletonJoints.labels[j])]
    )
  skeleton = np.stack(skeleton, 0)
  return skeleton


def skeleton_to_mano_min_hand(skeleton):
  """
  Map data from SkeletonJoints order to MANOHandJoints order.

  Parameters
  ----------
  skeleton : np.ndarray, [21, ...]
    Data in SkeletonJoints order. Note that the joints are along axis 0.

  Returns
  -------
  np.ndarray
    Data in MANOHandJoints order.
  """
  mano = []
  for j in range(MANOHandJoints.n_joints):
    mano.append(
      skeleton[SkeletonJointsMinHand.labels.index(MANOHandJoints.labels[j])]
    )
  mano = np.stack(mano, 0)
  return mano


def mano_to_skeleton_min_hand(mano):
  """
  Map data from MANOHandJoints order to SkeletonJoints order.

  Parameters
  ----------
  mano : np.ndarray, [21, ...]
    Data in MANOHandJoints order. Note that the joints are along axis 0.

  Returns
  -------
  np.ndarray
    Data in SkeletonJoints order.
  """
  skeleton = []
  for j in range(len(mano)):
    skeleton.append(
      mano[MANOHandJoints.labels.index(SkeletonJointsMinHand.labels[j])]
    )
  skeleton = np.stack(skeleton, 0)
  return skeleton
  

def xyz_to_delta(xyz, joints_def):
  """
  Compute bone orientations from joint coordinates (child joint - parent joint).
  The returned vectors are normalized.
  For the root joint, it will be a zero vector.

  Parameters
  ----------
  xyz : np.ndarray, shape [J, 3]
    Joint coordinates.
  joints_def : object
    An object that defines the kinematic skeleton, e.g. MPIIHandJoints.

  Returns
  -------
  np.ndarray, shape [J, 3]
    The **unit** vectors from each child joint to its parent joint.
    For the root joint, it's are zero vector.
  np.ndarray, shape [J, 1]
    The length of each bone (from child joint to parent joint).
    For the root joint, it's zero.
  """
  delta = []
  for j in range(joints_def.n_joints):
    p = joints_def.parents[j]
    if p is None:
      delta.append(np.zeros(3))
    else:
      delta.append(xyz[j] - xyz[p])
  delta = np.stack(delta, 0)
  lengths = np.linalg.norm(delta, axis=-1, keepdims=True)
  delta /= np.maximum(lengths, np.finfo(xyz.dtype).eps)
  return delta, lengths


def kypt_to_min_hand(kypt_xyz):
  # kypt->mano->mpii, rel to M0, normed by |M0 - Wrist| (configuration of min hand est)

  # kypt order -> mpii order
  mpii_xyz = mano_to_mpii(skeleton_to_mano_real(kypt_xyz))
  #mpii_xyz = skeleton_to_mano(kypt_xyz)  # worse meshes

  #print(mpii_xyz[9])

  # relative to wrist -> relative to M0
  rel_xyz = mpii_xyz - mpii_xyz[9]
  
  # distance between wrist (mpii_xyz[0]) and M0 (mpii_xyz[9])
  squared_dist = np.sum((mpii_xyz[9]-mpii_xyz[0])**2, axis=0)
  norm_dist = np.sqrt(squared_dist)  # / 2
  
  # normalize coordinates relative to M0 by calculated distance
  min_dist = 0
  norm_rel_xyz = (rel_xyz - min_dist) / (norm_dist - min_dist)  # norm_rel_xyz = rel_xyz / norm_dist

  return norm_rel_xyz, mpii_xyz[9]
