import numpy as np
import open3d
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
import os
import copy
import cv2
import pickle


def make_mesh(vertices, faces, mesh_2_world=None, rgba=[145, 114, 255, 255]):
    material = open3d.visualization.rendering.MaterialRecord()
    color = np.array(rgba) / 255.0
    material.base_color = color
    material.shader = "defaultLit"
    mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(vertices),
        open3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color[:3])

    if mesh_2_world is not None:
        mesh.transform(mesh_2_world)

    return mesh


def make_mano_mesh(vertices, faces, mesh_2_world=None, handVertContact=None, handVertIntersec=None):
    rgba = [245, 214, 175, 255]
    mesh = make_mesh(vertices,faces,mesh_2_world, rgba)

    vertex_colors = np.zeros((778, 3))

    color = np.array([0.960784314, 0.839215686, 0.68627451])

    vertex_colors[:, :] = color

    if handVertContact is not None:
        vertex_colors[:, 2] = handVertContact
        mesh.vertex_colors = open3d.utility.Vector3dVector(vertex_colors)

    if handVertIntersec is not None:
        vertex_colors[:, 0] = handVertIntersec
        mesh.vertex_colors = open3d.utility.Vector3dVector(vertex_colors)

    return mesh


def make_joint_sphere(joint):
    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=20)
    open3d.geometry.LineSet
    sphere.paint_uniform_color([0.8, 0.3, 0.3])
    sphere.translate(joint)
    sphere.compute_vertex_normals()
    sphere.compute_triangle_normals()
    return sphere


def recompute_mano_hand_state(joint_rotations, hand_state_left=None, hand_state_right=None, mano_shape_betas=None, left=True, mesh2world=None,
                                  handVertContact=None, handVertIntersec=None):
    if left:
        hand_state = hand_state_left
        color = [22, 33, 44, 255]
    else:
        hand_state = hand_state_right
        color = [145, 114, 240, 255]

    if mano_shape_betas is not None:
        hand_state.betas = mano_shape_betas

    hand_state.pose = np.ravel(joint_rotations[:16])
    hand_state.recompute_shape()
    hand_state.recompute_mesh(mesh2world=mesh2world)
    # hand_state.vertices = hand_state.vertices * 3

    mesh = make_mano_mesh(
        hand_state.vertices,
        hand_state.faces,
        handVertContact=handVertContact,
        handVertIntersec=handVertIntersec
    )

    return mesh


def mesh_mano(hand_state_left=None, hand_state_right=None, mesh_2_world=None, mano_shape_betas=None, mano_rotations_left=None,raw_3d_keypoints=None,
                    mano_rotations_right=None, handVertContact=None, handVertIntersec=None,handVertObj_vertices_in_world=None):

    # Calculate MANO Right Geometries
    mano_mesh_right = recompute_mano_hand_state(
        mano_rotations_right,
        hand_state_left, 
        hand_state_right,
        left=False,
        mano_shape_betas=mano_shape_betas,
        handVertContact=handVertContact,
        handVertIntersec=handVertIntersec,
        mesh2world=pt.concat(
            mano_2_world_base(hand_state_left, hand_state_right, mano_shape_betas=mano_shape_betas, left=False),
            mesh_2_world
        )
    )
    
    return mano_mesh_right


def mano_2_world_base(hand_state_left=None, hand_state_right=None, mano_shape_betas=None, left=True, factor_plus=1):
    mano_2_base = get_mano_base_joint(hand_state_left, hand_state_right, mano_shape_betas=mano_shape_betas, left=left)
    factor = -1

    factor = factor * factor_plus
    t = pt.transform_from(zero_rotations(),
                          [mano_2_base[0] * factor,
                           mano_2_base[1] * factor,
                           mano_2_base[2] * factor])
    return t


def get_mano_base_joint(hand_state_left=None, hand_state_right=None, mano_shape_betas=None, left=True):
    return get_mano_base_joints(hand_state_left, hand_state_right, mano_shape_betas=mano_shape_betas, left=left)[0]


def get_mano_base_joints(hand_state_left=None, hand_state_right=None, mano_shape_betas=None, left=True):
    if left:
        hand_state = hand_state_left
    else:
        hand_state = hand_state_right
    zero_rotations = np.zeros(shape=(16, 3))
    hand_state.pose = np.ravel(zero_rotations)
    hand_state.betas = mano_shape_betas
    hand_state.recompute_shape()
    hand_state.recompute_mesh(mesh2world=None)
    joints_xyz = hand_state.get_current_joints()
    return joints_xyz


def zero_rotations():
    return pr.matrix_from_compact_axis_angle([0, 0, 0])  # Rotation Matrix


def create_hand_mesh(res, hand_state_left, hand_state_right, mano_2_world_transform=np.eye(4)):

    # Calculate MANO2World
    mano_2_world = pt.concat(pt.transform_from(pr.matrix_from_compact_axis_angle([0, 0, 0]),
                                               res['world_xyz']
                                              ), mano_2_world_transform  # self.transformation_settings['mano_2_world']
                            )    
    
    real_mano_mesh_right = mesh_mano(  # aus winkeln
        hand_state_left=hand_state_left, 
        hand_state_right=hand_state_right,
        mesh_2_world=mano_2_world,
        mano_shape_betas=res["mano_shape"],
        mano_rotations_left=res['mano_joint_angles'],
        mano_rotations_right=res['mano_joint_angles'],
        handVertContact=res["handVertContact"],
        handVertIntersec=res["handVertIntersec"],
        raw_3d_keypoints=res['mpii_3d_joints'],
        handVertObj_vertices_in_world=None
    )  # green

    return real_mano_mesh_right


def create_object_mesh(res, hand_state_left, hand_state_right, mano_2_world_transform=np.eye(4)):

    objRot_angles = np.array([np.ndarray.item(res['objRot'][0]), np.ndarray.item(res['objRot'][1]), np.ndarray.item(res['objRot'][2])])

    if "mano_shape" in res:
        mano_shape = res["mano_shape"]
    else:
        mano_shape = np.zeros(10)

    if "mano_joint_angles" in res:
        mano_wrist_joint_angles = res['mano_joint_angles'][0]
    else:
        mano_wrist_joint_angles = np.zeros(shape=(16, 1))

    object_2_world = get_object_2_world(                                                      
        hand_state_left, 
        hand_state_right,
        objRot=objRot_angles,
        objTrans=res['objTrans'],
        mano_shape=mano_shape,
        mano_world_xyz=res['world_xyz'],
        mano_wrist_joint_angles=mano_wrist_joint_angles,
        mano_2_world_transform=mano_2_world_transform,  # self.transformation_settings['mano_2_world'],     
        wrist_zero_translation=False,  # self.general_settings['wrist_zero_translation']['selected'],
        wrist_zero_rotation=False  # self.general_settings['wrist_zero_rotation']['selected']
    )

    if "handVertObjSurfProj" in res:
        handVertObjSurfProj = res['handVertObjSurfProj']
    else:
        handVertObjSurfProj = np.zeros(shape=(778, 3))

    object_mesh, handVertObjSurfProj_pointcloud = mesh_object(res["objName"], res['objRot'], res['objTrans'], 
                                                              handVertObjSurfProj, 
                                                              mesh_2_world=object_2_world)
    
    return object_mesh


def mesh_object(objName, objRot, objTrans, handVertObjSurfProj, mesh_2_world=None):
    # load object model

    YCBModelsDir = 'sources/YCB_Video_Models'  # os.path.join('data', 'YCB_Video_Models')

    print("load obj: {}", objName)
    objMesh = read_obj(os.path.join(YCBModelsDir, 'models', objName, 'textured_simple.obj'))

    object_mesh_open3d = make_mesh(
        objMesh.v,
        objMesh.f)

    objMesh = object_mesh_open3d

    mesh_r = copy.deepcopy(objMesh)
    mesh_r.transform(mesh_2_world)

    handVertObjSurfProj_pointcloud = make_point_cloud(handVertObjSurfProj)

    #transform point clud by object translation+ object rotation
    handVertObjSurfProj_pointcloud.transform(
        pt.invert_transform(
            pt.transform_from(
                pr.matrix_from_compact_axis_angle(np.array(
        [np.ndarray.item(objRot[0]), np.ndarray.item(objRot[1]),
            np.ndarray.item(objRot[2])])),
                objTrans
            )
        )
    )
    #transform pointcloud with mesh_2_world
    handVertObjSurfProj_pointcloud.transform(mesh_2_world)

    return mesh_r , handVertObjSurfProj_pointcloud


def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': [], 'fn': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0]) - 1 for l in spl[:3]], dtype=np.uint32)])
            if len(spl[0]) > 1 and spl[1] and 'ft' in d:
                d['ft'].append([np.array([int(l[1]) - 1 for l in spl[:3]])])
            if len(spl[0]) > 2 and spl[2] and 'fn' in d:
                d['fn'].append([np.array([int(l[2]) - 1 for l in spl[:3]])])

            # TOO: redirect to actual vert normals?
            # if len(line[0]) > 2 and line[0][2]:
            #    d['fn'].append([np.concatenate([l[2] for l in spl[:3]])])
        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])

    for k, v in d.items():
        if k in ['v', 'vn', 'f', 'vt', 'ft', 'fn']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result


class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs


def make_point_cloud(vertices, color=None):
    pc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(vertices))
    if color is not None:
        pc.paint_uniform_color(color)
    return pc


def zero_transformation():
    # Transform and Rotate to World
    r = pr.matrix_from_compact_axis_angle([0, 0, 0])  # Rotation Matrix
    t = pt.transform_from(r, [0, 0, 0])  # Translation
    return t


def get_object_2_world(hand_state_left, 
                       hand_state_right,
                       mano_2_world_transform=None,
                       mano_world_xyz=np.zeros(3),
                       mano_wrist_joint_angles=np.zeros(3),
                       mano_shape=np.zeros(10),
                       objRot=np.zeros(3),
                       objTrans=np.zeros(3),
                       wrist_zero_translation=False,
                       wrist_zero_rotation=False
                       ):
    
    t = zero_transformation()

    t = pt.concat(
        zero_transformation(),
        pt.transform_from(
            pr.matrix_from_compact_axis_angle(objRot),
            [0, 0, 0]
        ),
    )
    t = pt.concat(
        t,
        pt.transform_from(
            zero_rotations(),
            objTrans
        ),
    )

    t = pt.concat(
        t,
        mano_2_world_base(hand_state_left, hand_state_right, mano_shape_betas=mano_shape, left=False),

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
                get_mano_orientation_transformation(
                    mano_wrist_joint_angles=mano_wrist_joint_angles)
            ),
        )
    t = pt.concat(
        t,
        mano_2_world_transform
    )
    return t


def get_mano_orientation_transformation(mano_wrist_joint_angles=np.zeros(3)):
    mano_orientation = pt.transform_from(
        pr.matrix_from_compact_axis_angle(mano_wrist_joint_angles),
        [0, 0, 0]
    )
    return mano_orientation


def load_annotations_and_image(seq_dir, frame_number, eval=False):

    rgb_filename = os.path.join(seq_dir, 'rgb', str(frame_number).zfill(4) + ".jpg")
    rgb_img = cv2.imread(rgb_filename)

    meta_filename = os.path.join(seq_dir, 'meta', str(frame_number).zfill(4) + '.pkl')
    metafilename = open(meta_filename, 'rb')
    ho3d_annotation = pickle.load(metafilename, encoding='latin1')
    #ho3d_annotation = pickle.load(metafilename)

    if eval is False:
        res = {'world_xyz': ho3d_annotation['handTrans'],
            'mano_joint_angles': ho3d_annotation['handPose'].reshape((16, 3)),
            'mininimal_hand_absolute_joint_rotations': np.zeros(shape=(21, 3)),
            'annotated_image': rgb_img,
            'mpii_3d_joints': None,
            'handVertContact': ho3d_annotation['handVertContact'],
            'handVertIntersec': ho3d_annotation["handVertIntersec"],
            'mano_shape': ho3d_annotation['handBeta'],
            'objTrans': ho3d_annotation['objTrans'],
            'objRot': ho3d_annotation['objRot'],
            'objName': ho3d_annotation['objName'],
            'objLabel': ho3d_annotation['objLabel'],
            'handVertObjSurfProj': ho3d_annotation['handVertObjSurfProj']
            }
    else:
        
        res = {
            'objTrans': ho3d_annotation['objTrans'],
            'objRot': ho3d_annotation['objRot'],
            'objName': ho3d_annotation['objName'],
            'objLabel': ho3d_annotation['objLabel'],
            'world_xyz' : ho3d_annotation['handJoints3D']
            }
        """
        # h203d
        res = {
            'objTrans': ho3d_annotation['objTrans'],
            'objRot': ho3d_annotation['objRot'],
            'objName': ho3d_annotation['objName'],
            'world_xyz' : ho3d_annotation['rightHandJoints3D']
            }
        """
    return res


def load_annotations_and_image_eval(seq_dir, frame_number):

    rgb_filename = os.path.join(seq_dir, 'rgb', str(frame_number).zfill(4) + ".jpg")
    rgb_img = cv2.imread(rgb_filename)

    meta_filename = os.path.join(seq_dir, 'meta', str(frame_number).zfill(4) + '.pkl')
    metafilename = open(meta_filename, 'rb')
    ho3d_annotation = pickle.load(metafilename, encoding='latin1')

    res = {'world_xyz': np.zeros(shape=(1, 3)),
           'mano_joint_angles': np.zeros(shape=(16, 3)),
           'mano_shape': np.zeros(shape=(10, 1)),
           'objTrans': ho3d_annotation['objTrans'],
           'objRot': ho3d_annotation['objRot'],
           'objName': ho3d_annotation['objName'],
           'handVertObjSurfProj': np.zeros(shape=(778, 3))
           }

    handJoints3D = ho3d_annotation['handJoints3D']

    return res, handJoints3D
