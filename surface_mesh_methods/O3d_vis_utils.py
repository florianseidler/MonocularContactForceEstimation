import open3d as o3d
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr

from .hand_mesh.kinematics import MANOHandJoints


class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

class O3dVisUtils:


    @staticmethod
    def make_mesh(vertices, faces, mesh_2_world=None, rgba=[145, 114, 255, 255]):
        material = o3d.visualization.rendering.MaterialRecord()
        # color = np.array(rgba) / 255.0
        color = np.array([13, 114, 175, 255]) / 255.0  # blue
        material.base_color = color
        material.shader = "defaultLit"
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color[:3])

        if mesh_2_world is not None:
            mesh.transform(mesh_2_world)

        return mesh

    @staticmethod
    def make_mano_mesh(vertices, faces, mesh_2_world=None, handVertContact=None, handVertIntersec=None):
        rgba = [245, 214, 175, 255]
        mesh = O3dVisUtils.make_mesh(vertices,faces,mesh_2_world, rgba)

        vertex_colors = np.zeros((778, 3))

        color = np.array([0.960784314, 0.839215686, 0.68627451])

        vertex_colors[:, :] = color

        if handVertContact is not None:
            vertex_colors[:, 2] = handVertContact
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        if handVertIntersec is not None:
            vertex_colors[:, 0] = handVertIntersec
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        return mesh

    @staticmethod
    def get_color_by_joint_index(joint_index):
        options = {0: [0, 0, 0],
                   1: [1, 0, 0],
                   2: [1, 0, 0],
                   3: [1, 0, 0],
                   4: [0, 1, 0],
                   5: [0, 1, 0],
                   6: [0, 1, 0],
                   7: [0, 0, 1],
                   8: [0, 0, 1],
                   9: [0, 0, 1],
                   10: [0, 1, 1],
                   11: [0, 1, 1],
                   12: [0, 1, 1],
                   13: [1, 0, 1],
                   14: [1, 0, 1],
                   15: [1, 0, 1],
                   16: [1, 1, 0],
                   17: [1, 1, 0],
                   18: [1, 1, 0],
                   19: [1, 1, 0],
                   20: [1, 1, 0],
                   }
        if 0 <= joint_index <= 20:
            color = options[joint_index]
        else:
            color = [0.9, 1, 0.7]
        return color


    @staticmethod
    def make_finger_sceleton_with_coords(joints_xyz, joint_rotations):
        geometries = O3dVisUtils.make_finger_sceleton(joints_xyz)

        joint_rotations_new = np.zeros(shape=(16,3))
        for j in range(16):
            joint_rotations_new[j] = joint_rotations[j]
            parent = MANOHandJoints.parents[j]
            if parent is not None:
                joint_rotations_new[j] += joint_rotations_new[parent]

        for idx, pose in enumerate(joint_rotations_new):
            coord = O3dVisUtils.make_coord_frame(size=0.03)
            coord.translate(joints_xyz[idx])
            coord.rotate(pr.matrix_from_compact_axis_angle(pose))
            geometries.append(coord)

        return geometries

    @staticmethod
    def make_finger_sceleton(joints_xyz, mesh_2_world=None):

        geometries = []
        for idx, pos in enumerate(joints_xyz):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001, resolution=20)
            colors = np.zeros((len(joints_xyz), 3))
            colors[:] = (0.8, 0.3, 0.3)
            sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
            sphere.translate(pos)

            if mesh_2_world is not None:
                sphere.transform(mesh_2_world)

            sphere.compute_vertex_normals()
            sphere.compute_triangle_normals()

            geometries.append(sphere)

        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[0], joints_xyz[1],
                                                       color=O3dVisUtils.get_color_by_joint_index(1)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[1], joints_xyz[2],
                                                       color=O3dVisUtils.get_color_by_joint_index(2)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[2], joints_xyz[3],
                                                       color=O3dVisUtils.get_color_by_joint_index(3)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[0], joints_xyz[4],
                                                       color=O3dVisUtils.get_color_by_joint_index(4)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[4], joints_xyz[5],
                                                       color=O3dVisUtils.get_color_by_joint_index(5)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[5], joints_xyz[6],
                                                       color=O3dVisUtils.get_color_by_joint_index(6)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[0], joints_xyz[7],
                                                       color=O3dVisUtils.get_color_by_joint_index(7)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[7], joints_xyz[8],
                                                       color=O3dVisUtils.get_color_by_joint_index(8)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[8], joints_xyz[9],
                                                       color=O3dVisUtils.get_color_by_joint_index(9)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[0], joints_xyz[10],
                                                       color=O3dVisUtils.get_color_by_joint_index(10)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[10], joints_xyz[11],
                                                       color=O3dVisUtils.get_color_by_joint_index(11)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[11], joints_xyz[12],
                                                       color=O3dVisUtils.get_color_by_joint_index(12)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[0], joints_xyz[13],
                                                       color=O3dVisUtils.get_color_by_joint_index(13)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[13], joints_xyz[14],
                                                       color=O3dVisUtils.get_color_by_joint_index(14)))
        geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[14], joints_xyz[15],
                                                       color=O3dVisUtils.get_color_by_joint_index(15)))


        if len(joints_xyz) >= 21:
            geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[3], joints_xyz[16],
                                                           color=O3dVisUtils.get_color_by_joint_index(3)))
            geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[6], joints_xyz[17],
                                                           color=O3dVisUtils.get_color_by_joint_index(6)))
            geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[9], joints_xyz[18],
                                                           color=O3dVisUtils.get_color_by_joint_index(9)))
            geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[12], joints_xyz[19],
                                                           color=O3dVisUtils.get_color_by_joint_index(12)))
            geometries.append(O3dVisUtils.make_joint_arrow(joints_xyz[15], joints_xyz[20],
                                                           color=O3dVisUtils.get_color_by_joint_index(15)))
        return geometries

    """"
    Draw spheres at given positions
    
    Parameters
    ----------
    joints_xyz : array of length n, shape (n, 3)
        Joint positions
    
    name : String
        Unique name as open3D rendered object id (important for refreshment)    
    """

    @staticmethod
    def make_joint_spheres(joints_xyz):
        spheres = []
        for idx, pos in enumerate(joints_xyz):
            spheres.append(O3dVisUtils.make_joint_sphere(pos))
        return spheres

    @staticmethod
    def make_joint_sphere(joint):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=20)
        o3d.geometry.LineSet
        sphere.paint_uniform_color([0.8, 0.3, 0.3])
        sphere.translate(joint)
        sphere.compute_vertex_normals()
        sphere.compute_triangle_normals()
        return sphere


    @staticmethod
    def make_coord_frame(size=1):
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    """
    Warning this LineSet generates a Warning message, everytime is is rendered!
    Maybe use instead: o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)?
    #"""

    @staticmethod
    def make_coordinate_system(s, short_tick_length=0.01, long_tick_length=0.05):
        coordinate_system = o3d.geometry.LineSet()
        points = []
        lines = []
        colors = []
        for d in range(3):
            color = [0, 0, 0]
            color[d] = 1

            start = [0, 0, 0]
            start[d] = -s
            end = [0, 0, 0]
            end[d] = s

            points.extend([start, end])
            lines.append([len(points) - 2, len(points) - 1])
            colors.append(color)
            for i, step in enumerate(np.arange(-s, s + 0.01, 0.01)):
                tick_length = long_tick_length if i % 5 == 0 else short_tick_length
                start = [0, 0, 0]
                start[d] = step
                start[(d + 2) % 3] = -tick_length
                end = [0, 0, 0]
                end[d] = step
                end[(d + 2) % 3] = tick_length
                points.extend([start, end])
                lines.append([len(points) - 2, len(points) - 1])
                colors.append(color)
            coordinate_system.points = o3d.utility.Vector3dVector(points)
            coordinate_system.lines = o3d.utility.Vector2iVector(lines)
            coordinate_system.colors = o3d.utility.Vector3dVector(colors)
        return coordinate_system

    @staticmethod
    def make_joint_arrows(joints, mesh_2_world=None):
        i = 0
        geometries = []
        for joint in joints:
            if not (joint[0] == 0 and joint[1] == 0 and joint[2] == 0):
                geometry = O3dVisUtils.make_joint_arrow([0, 0, 0], joint, color=O3dVisUtils.get_color_by_joint_index(i))
                if mesh_2_world is not None:
                    geometry.transform(mesh_2_world)
                geometries.append(geometry)

                i = i + 1
        return geometries



    @staticmethod
    def make_speres(joints,radius=0.1):
        i = 0
        geometries = []
        for idx,joint in enumerate(joints):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(joint)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(O3dVisUtils.get_color_by_joint_index(idx))

            geometries.append(sphere)
            i = i + 1
        return geometries

    @staticmethod
    def make_point_cloud(vertices, color=None):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
        if color is not None:
            pc.paint_uniform_color(color)
        return pc


    @staticmethod
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

    @staticmethod
    def create_transformation_frame(transformation, mesh_2_world=None,color=None, size=0.02):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        mesh.transform(transformation)
        if mesh_2_world is not None:
            mesh.transform(mesh_2_world)
        return mesh

    @staticmethod
    def create_colored_hand_sceleton(joints, all_colors=None, uncolored=False, aboslute_joints=False):

        if all_colors is not None:
            scale_colors = all_colors['mano_joint_scale_colors']  # 21
            pose_colors = all_colors['mano_pose_offset']  # 21
            joint_colors = all_colors['mano_joints_offset']  # 21

        geometries = []
        """
        for idx,t in enumerate(transformations):
            mesh = O3dVisUtils.create_transformation_frame(t, mesh_2_world=mesh_2_world,size=0.013)
            mesh.paint_uniform_color([pose_colors[idx],1-pose_colors[idx],0])
            mesh.compute_vertex_normals()
            geometries.append(mesh)
        """

        for idx, joint in enumerate(joints):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
            sphere.translate(joint)
            if uncolored:
                sphere.paint_uniform_color(O3dVisUtils.get_color_by_joint_index(idx))
            else:
                color = min(1, joint_colors[idx] + pose_colors[idx])
                sphere.paint_uniform_color([color, 1 - color, 0])

            sphere.compute_vertex_normals()
            geometries.append(sphere)

        def add_bone(index_a, index_b, size_factor=1):

            if uncolored:
                color = [0.5, 0.5, 0.5]
            else:
                color = [scale_colors[index_a], 1 - scale_colors[index_a], 0]

            g = O3dVisUtils.make_joint_bone(
                joints[index_a],
                joints[index_b],
                color=color,
                size_factor=size_factor
            )
            geometries.append(g)

        if not aboslute_joints:
            add_bone(0, 1, size_factor=0.5)
            add_bone(1, 2)
            add_bone(2, 3)
            add_bone(0, 4, size_factor=0.5)
            add_bone(4, 5)
            add_bone(5, 6)
            add_bone(0, 7, size_factor=0.5)
            add_bone(7, 8)
            add_bone(8, 9)
            add_bone(0, 10, size_factor=0.5)
            add_bone(10, 11)
            add_bone(11, 12)
            add_bone(0, 13, size_factor=0.5)
            add_bone(13, 14)
            add_bone(14, 15)
            add_bone(3, 16)
            add_bone(6, 17)
            add_bone(9, 18)
            add_bone(12, 19)
            add_bone(15, 20)
        else:
            add_bone(0, 1, size_factor=0.2)
            add_bone(0, 2, size_factor=0.2)
            add_bone(0, 3, size_factor=0.2)
            add_bone(0, 4, size_factor=0.2)
            add_bone(0, 5, size_factor=0.2)
            add_bone(0, 6, size_factor=0.2)
            add_bone(0, 7, size_factor=0.2)
            add_bone(0, 8, size_factor=0.2)
            add_bone(0, 9, size_factor=0.2)
            add_bone(0, 10, size_factor=0.2)
            add_bone(0, 11, size_factor=0.2)
            add_bone(0, 12, size_factor=0.2)
            add_bone(0, 13, size_factor=0.2)
            add_bone(0, 14, size_factor=0.2)
            add_bone(0, 15, size_factor=0.2)
            add_bone(0, 16, size_factor=0.2)
            add_bone(0, 17, size_factor=0.2)
            add_bone(0, 18, size_factor=0.2)
            add_bone(0, 19, size_factor=0.2)
            add_bone(0, 20, size_factor=0.2)

        return geometries


    @staticmethod
    def create_colored_hand_mesh(transformations, all_colors=None, uncolored=False):
        all_joints = []
        for idx, t in enumerate(transformations):
            pq = pt.pq_from_transform(t)
            joint = [pq[0], pq[1], pq[2]]
            all_joints.append(joint)
        return O3dVisUtils.create_colored_hand_sceleton(all_joints, all_colors=all_colors,uncolored=uncolored)

    @staticmethod
    def create_colored_hand_sceleton_transformations(transformations, all_colors=None, uncolored=False):
        all_joints = []
        for idx, t in enumerate(transformations):
            pq = pt.pq_from_transform(t)
            joint = [pq[0], pq[1], pq[2]]
            all_joints.append(joint)
        return O3dVisUtils.create_colored_hand_sceleton(all_joints, all_colors=all_colors,uncolored=uncolored)

    @staticmethod
    def create_transformation_frames(transformations, mesh_2_world=None, colors=None):
        geometries = []
        for t in transformations:
            mesh = O3dVisUtils.create_transformation_frame(t, mesh_2_world=mesh_2_world,size=0.015)

            mesh.paint_uniform_color([0,1,0])
            mesh.compute_vertex_normals()

            geometries.append(mesh)
        return geometries


    @staticmethod
    def make_joint_bone(joint_a, joint_b, color=[1, 0, 1],size_factor=1):
        vec_Arr = np.array(joint_b) - np.array(joint_a)
        vec_len = np.linalg.norm(vec_Arr)

        if vec_len == 0.0:
            vec_Arr = np.array(joint_b) - (np.array(joint_a)+0.00000001)

        mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.00001,
            cone_radius=0.00001,
            cylinder_height=0.9999,
            cylinder_radius=0.04*size_factor
        )
        mesh_arrow.paint_uniform_color(color)
        mesh_arrow.compute_vertex_normals()
        rot_mat = O3dVisUtils.calculate_align_mat(vec_Arr)
        mesh_arrow.translate(np.array(joint_a))  # 0.5*(np.array(end) - np.array(begin))
        mesh_arrow.rotate(rot_mat, center=joint_a)
        return mesh_arrow

    @staticmethod
    def make_joint_arrow(joint_a, joint_b, color=[1, 0, 1],size=1):
        vec_Arr = np.array(joint_b) - np.array(joint_a)
        vec_len = np.linalg.norm(vec_Arr)

        if vec_len == 0.0:
            vec_Arr = np.array(joint_b) - (np.array(joint_a)+0.00000001)

        mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.2,
            cone_radius=0.08,
            cylinder_height=0.8,
            cylinder_radius=0.02
        )
        mesh_arrow.paint_uniform_color(color)
        mesh_arrow.compute_vertex_normals()
        rot_mat = O3dVisUtils.calculate_align_mat(vec_Arr)
        mesh_arrow.translate(np.array(joint_a))  # 0.5*(np.array(end) - np.array(begin))
        mesh_arrow.rotate(rot_mat, center=joint_a)
        return mesh_arrow

    @staticmethod
    def get_cross_prod_mat(pVec_Arr):
        # pVec_Arr shape (3)
        qCross_prod_mat = np.array([
            [0, -pVec_Arr[2], pVec_Arr[1]],
            [pVec_Arr[2], 0, -pVec_Arr[0]],
            [-pVec_Arr[1], pVec_Arr[0], 0],
        ])
        return qCross_prod_mat

    @staticmethod
    def calculate_align_mat(pVec_Arr):
        scale = np.linalg.norm(pVec_Arr)
        pVec_Arr = pVec_Arr / scale
        # must ensure pVec_Arr is also a unit vec.
        z_unit_Arr = np.array([0, 0, 1])
        z_mat = O3dVisUtils.get_cross_prod_mat(z_unit_Arr)

        z_c_vec = np.matmul(z_mat, pVec_Arr)
        z_c_vec_mat = O3dVisUtils.get_cross_prod_mat(z_c_vec)

        if np.dot(z_unit_Arr, pVec_Arr) == -1:
            qTrans_Mat = -np.eye(3, 3)
        elif np.dot(z_unit_Arr, pVec_Arr) == 1:
            qTrans_Mat = np.eye(3, 3)
        else:
            qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                                z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))

        qTrans_Mat *= scale
        return qTrans_Mat