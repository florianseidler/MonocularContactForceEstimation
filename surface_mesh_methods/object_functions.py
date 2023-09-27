import torch
import numpy as np
from copy import deepcopy
import cv2
import open3d
import os


def kypt_demo_obj_mesh_todel(obj_mesh, obj_rot, obj_trans):
    #obj_id = int(meta_info['obj_id'][0].cpu().numpy())
    #ycb_objs = H2O3DObjects()
    #from copy import deepcopy
    #obj_mesh = deepcopy(ycb_objs.obj_id_to_mesh[obj_id])

    obj_rot = torch.from_numpy(obj_rot)
    obj_trans = torch.from_numpy(obj_trans)
    # Get the object mesh
    rot_z_mat = np.eye(4)
    rot_z_mat[:3,:3] = cv2.Rodrigues(np.array([0,1,0])*np.pi)[0]

    use_obj_rot_parameterization = 1
    if use_obj_rot_parameterization:  # cfg.use_obj_rot_parameterization:
        #rot_mat = rot_param_rot_mat(obj_rot[0:0 + 1].reshape(-1, 6))[0].cpu().numpy()  # 3 x 3
        rot_mat = rot_param_rot_mat(obj_rot.reshape(-1, 6))[0].cpu().numpy()  # 3 x 3
    else:
        #rot_mat = cv2.Rodrigues(obj_rot[0])[0]  # 
        # for a 3-element vector that encodes the axis of rotation and the angle of rotation
        rot_mat = cv2.Rodrigues(obj_rot[0].cpu().numpy())[0]

    trans_mat = np.eye(4)
    trans_mat[:3,:3] = rot_mat#.dot(aa)
    trans_mat[:3,3] = obj_trans[0].cpu().numpy()
    print(trans_mat[2,3] )
    #trans_mat[2,3] = -trans_mat[2,3]
    print(trans_mat[2,3] )

    #rad = np.pi/2
    #rad = 3 * np.pi/4
    rad = np.pi
    #obj_mesh = rotated_mesh_y(obj_mesh, rad)  # obj_mesh_y
    #obj_mesh = rotated_mesh_z(obj_mesh, rad)  # obj_mesh_z
    #obj_mesh.transform(rot_mat)

    obj_mesh = rotated_mesh_x(obj_mesh, rad)  # obj_mesh_x
    #obj_mesh = rotated_mesh_z(obj_mesh, rad=np.pi/2)  # obj_mesh_x
    obj_mesh.transform(rot_z_mat)
    obj_mesh.transform(trans_mat)
    rad = -np.pi/2
    #obj_mesh_x, obj_mesh_y, obj_mesh_z = rotated_meshes(obj_mesh, rad)
    #obj_mesh_x.transform(trans_mat)
    #obj_mesh_y.transform(trans_mat)
    #obj_mesh_z.transform(trans_mat)

    return obj_mesh


def kypt_demo_obj_mesh(obj_mesh, obj_rot, obj_trans):

    obj_rot = torch.from_numpy(obj_rot)
    obj_trans = torch.from_numpy(obj_trans)
    # Get the object mesh
    rot_z_mat = np.eye(4)
    rot_z_mat[:3,:3] = cv2.Rodrigues(np.array([0,1,0])*np.pi)[0]

    use_obj_rot_parameterization = 1
    if use_obj_rot_parameterization:  # cfg.use_obj_rot_parameterization:
        rot_mat = rot_param_rot_mat(obj_rot.reshape(-1, 6))[0].cpu().numpy()  # 3 x 3
    else:
        # for a 3-element vector that encodes the axis of rotation and the angle of rotation
        rot_mat = cv2.Rodrigues(obj_rot[0].cpu().numpy())[0]

    trans_mat = np.eye(4)
    trans_mat[:3,:3] = rot_mat#.dot(aa)
    trans_mat[:3,3] = obj_trans[0].cpu().numpy()

    obj_mesh = rotated_mesh_x(obj_mesh, rad=np.pi)
    obj_mesh.transform(rot_z_mat)
    obj_mesh.transform(trans_mat)

    return obj_mesh


def rotated_meshes(object_mesh, rad):
    rot_mat_x = rot_x(rad)
    rot_mat_y = rot_y(rad)
    rot_mat_z = rot_z(rad)
    object_color_x = np.array([213, 14, 175,  55]) / 255.0  # magenta
    object_color_y = np.array([113, 214, 75, 155]) / 255.0  # light green
    object_color_z = np.array([13, 114, 105, 205]) / 255.0  # dark green
    obj_mesh_x = copy_trafo_color(object_mesh, rot_mat_x, object_color_x)
    obj_mesh_y = copy_trafo_color(object_mesh, rot_mat_y, object_color_y)
    obj_mesh_z = copy_trafo_color(object_mesh, rot_mat_z, object_color_z)
    return obj_mesh_x, obj_mesh_y, obj_mesh_z


def rotated_mesh_x(object_mesh, rad):
    rot_mat_x = rot_x(rad)
    object_color_x = np.array([213, 14, 175,  55]) / 255.0  # magenta
    obj_mesh_x = copy_trafo_color(object_mesh, rot_mat_x, object_color_x)
    return obj_mesh_x


def rotated_mesh_y(object_mesh, rad):
    rot_mat_y = rot_y(rad)
    object_color_y = np.array([113, 214, 75, 155]) / 255.0  # light green
    obj_mesh_y = copy_trafo_color(object_mesh, rot_mat_y, object_color_y)
    return obj_mesh_y


def rotated_mesh_z(object_mesh, rad):
    rot_mat_z = rot_z(rad)
    object_color_z = np.array([13, 114, 105, 205]) / 255.0  # dark green
    obj_mesh_z = copy_trafo_color(object_mesh, rot_mat_z, object_color_z)
    return obj_mesh_z


def rot_param_rot_mat(rot):
    '''
    stacks three unit vectors along third dimension to create an N x 3 x 3 tensor 
    representing the rotation matrices for each input rotation vector
    -----------------------
    input: rot is an N x 6 tensor
        first three columns: first vector e1 
        last three columns: second vector e2
    return: rotation matrix R 3x3
    '''

    e1 = rot[:,:3]
    e2 = rot[:,3:]

    # normalizes e1 to get a unit vector e1d
    e1d = e1/torch.linalg.norm(e1,dim=1,keepdim=True)
    # calculates the cross product of e1d and e2 to get a third vector
    # which is normalized to get a unit vector e3d
    e3d = torch.cross(e1d, e2, dim=1)/torch.linalg.norm(e2,dim=1,keepdim=True)
    # cross product of e3d and e1d gives the second unit vector e2d
    e2d = torch.cross(e3d, e1d)

    # stack three unit vectors along third dimension
    R = torch.stack([e1d, e2d, e3d],dim=2) # N x 3 x 3

    return R


def rot_x(rad):
    rot_mat = np.eye(4)
    rot_mat[1,1] = np.cos(rad)
    rot_mat[2,2] = np.cos(rad)
    rot_mat[1,2] = np.sin(rad)
    rot_mat[2,1] = -np.sin(rad)
    return rot_mat


def rot_y(rad):
    rot_mat = np.eye(4)
    rot_mat[0,0] = np.cos(rad)
    rot_mat[2,2] = np.cos(rad)
    rot_mat[0,2] = np.sin(rad)
    rot_mat[2,0] = -np.sin(rad)
    return rot_mat


def rot_z(rad):
    rot_mat = np.eye(4)
    rot_mat[0,0] = np.cos(rad)
    rot_mat[1,1] = np.cos(rad)
    rot_mat[0,1] = -np.sin(rad)
    rot_mat[1,0] = np.sin(rad)
    return rot_mat


def copy_trafo_color(object_mesh, rot_mat, object_color):
    mesh = deepcopy(object_mesh)
    mesh.transform(rot_mat)
    mesh.paint_uniform_color(object_color[:3])
    return mesh


class H2O3DObjects():
    def __init__(self):
        self.get_obj_id_to_mesh()

    def get_obj_id_to_mesh(self):
        import keypoint_transformer.config as cfg
        import os
        YCB_models_dir = cfg.object_models_dir
        obj_names = os.listdir(YCB_models_dir)
        self.obj_id_to_name = {int(o[:3]):o for o in obj_names}
        self.obj_id_to_mesh = {}
        self.obj_id_to_dia = {}
        for id in self.obj_id_to_name.keys():
            if id not in [3,4,6,10,11,19,21,25,35,37,24]:
                continue
            obj_name = self.obj_id_to_name[id]
            # print(os.path.join(YCB_models_dir, obj_name, 'textured_simple_2000.obj'))
            assert os.path.exists(os.path.join(YCB_models_dir, obj_name, 'textured_simple.obj'))
            import open3d as o3d
            o3d_mesh = o3d.io.read_triangle_mesh(os.path.join(YCB_models_dir, obj_name, 'textured_simple.obj'))
            # o3d.visualization.draw_geometries([o3d_mesh])
            self.obj_id_to_mesh[id] = o3d_mesh


class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs


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


def untransformed_obj_mesh(objName):
    YCBModelsDir = 'sources/YCB_Video_Models'
    print("load obj: {}", objName)
    objMesh = read_obj(os.path.join(YCBModelsDir, 'models', objName, 'textured_simple.obj'))
    object_mesh_open3d = make_mesh(
        objMesh.v,
        objMesh.f)
    # add rotation if needed
    rot = 0
    if rot:
        rot_mat = np.eye(4)
        rot_mat[:3,:3] = cv2.Rodrigues(np.array([0,0,1])*np.pi)[0]
        object_mesh_open3d.transform(rot_mat)

    return object_mesh_open3d