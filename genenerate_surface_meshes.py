from keypoint_transformer.call_kypt_transformer import kypt_trafo
import open3d
import numpy as np
from surface_mesh_methods.object_functions import kypt_demo_obj_mesh, rotated_mesh_x
import os
# execute from dir: rgb_contact_force_estimation


set = 'GSF10'  # "AP11"  # "SM1"  # "MPM13"  # "MPM14"  # "SB11"  # "GSF10"  # "SMu40"


def main():
    mano_meshes, object_meshes, obj_translation_vec, obj_rotation_vec = kypt_trafo()

    # Create a vector to store file paths
    file_paths = []

    # set color
    hand_color = np.array([0.960784314, 0.839215686, 0.68627451])
    object_color = np.array([13, 114, 175, 255]) / 255.0  # blue

    # add color to meshes for visualisation and rotate if necessary
    rad = np.pi#/2
    for i in range(len(mano_meshes)):
        ##mano_meshes[i] = rotated_mesh_x(mano_meshes[i], rad)
        ##mano_meshes[i] = rotated_mesh_x(mano_meshes[i], rad)
        mano_meshes[i].paint_uniform_color(hand_color[:3])

    for i in range(len(object_meshes)):
        obj_rot = np.array(obj_rotation_vec[i]).reshape(1, 6)
        obj_trans = np.array(obj_translation_vec[i]).reshape(1, 3)
        object_meshes[i] = kypt_demo_obj_mesh(object_meshes[i], obj_rot, obj_trans)
        # rotate object mesh so that it fits to hand (passive rotation)
        object_meshes[i] = rotated_mesh_x(object_meshes[i], rad)
        object_meshes[i].paint_uniform_color(object_color[:3])

    # Define the directory path
    mano_base_dir = "output/" + set + "_meshes/" + set + f"_mano_meshes/"
    object_base_dir = "output/" + set + "_meshes/" + set + f"_object_meshes/"

    # Check if the directory exists, if not, create it
    if not os.path.isdir(mano_base_dir):
        os.makedirs(mano_base_dir)

    if not os.path.isdir(object_base_dir):
        os.makedirs(object_base_dir)

    # Save each mesh and store the file path in the vector
    for i, mesh in enumerate(mano_meshes):
        file_path = mano_base_dir + f"mano_mesh_{i}.stl" 
        open3d.io.write_triangle_mesh(file_path, mesh)
        file_paths.append(file_path)

    for i, mesh in enumerate(object_meshes):
        file_path = object_base_dir + f"object_mesh_{i}.stl"
        open3d.io.write_triangle_mesh(file_path, mesh)
        file_paths.append(file_path)

    # Print the file paths
    for file_path in file_paths:
        print(file_path)


if __name__ == "__main__":
    main()