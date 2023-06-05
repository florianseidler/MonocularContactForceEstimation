import numpy as np
import open3d
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
from methods import load_annotations_and_image, create_object_mesh, create_hand_mesh
from hand_pose_mesh.mano import HandState
from hand_pose_mesh.MANO_pose_generator import generate_mano_mesh_from_kypt
import os

base_path = os.getcwd()

trained_data = False 
set = 'SB13'
frame = '0618'

if trained_data:
    seq_dir = base_path + '/sources/HO3D_v3/train/' + set + '/'
else:  # if data is unseen
    seq_dir = base_path + '/sources/HO3D_v3/evaluation/' + set + '/


def main():   
    # load rotation and translation of object, translation of hand and annotated handpose if available
    res = load_annotations_and_image(seq_dir, frame, eval=True)
    
    hand_state_left = HandState(left=True)
    hand_state_right = HandState(left=False)
    mano_2_world_transform = np.eye(4)

    # create meshes
    mano_mesh_kypt = generate_mano_mesh_from_kypt(seq_dir + 'rgb/' + frame, seq_dir, frame)  # blue    
    mano_mesh_kypt.translate(res['world_xyz'])

    object_mesh = create_object_mesh(res, hand_state_left, hand_state_right, mano_2_world_transform)
    #real_mano_mesh = create_hand_mesh(res, hand_state_left, hand_state_right, mano_2_world_transform)
    coord_sys_mesh = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Visualize meshes
    open3d.visualization.draw_geometries([coord_sys_mesh, mano_mesh_kypt, object_mesh])  # , real_mano_mesh])


if __name__ == "__main__":
    main()
