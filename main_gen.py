from hand_pose_mesh.kypt_trafo import kypt_trafo_old
import open3d
import numpy as np


set = 'SM1'  # "AP11"  # "SM1"  # "MPM13"  # "MPM14"  # "SB11"  # "GSF10"  # "SMu40"

(mano_meshes, object_meshes, translation_vec, 
handJoints3D_vec, m0_translation_vec, realhandtrans_vec, 
obj_translation_vec, obj_rotation_vec, kypt_inv_trans_vec) = kypt_trafo_old()

# Create a vector to store file paths
file_paths = []

# set color
hand_color = np.array([0.960784314, 0.839215686, 0.68627451])
object_color = np.array([13, 114, 175, 255]) / 255.0  # blue

rad = np.pi#/2
from object_functions import rotated_mesh_x
for i in range(len(mano_meshes)):
    #mano_meshes[i] = rotated_mesh_x(mano_meshes[i], rad)
    #mano_meshes[i] = rotated_mesh_x(mano_meshes[i], rad)
    mano_meshes[i].paint_uniform_color(hand_color[:3])
from object_functions import kypt_demo_obj_mesh_old
for i in range(len(object_meshes)):
    obj_rot = np.array(obj_rotation_vec[i]).reshape(1, 6)
    obj_trans = np.array(obj_translation_vec[i]).reshape(1, 3)
    object_meshes[i] = kypt_demo_obj_mesh_old(object_meshes[i], obj_rot, obj_trans)
    object_meshes[i] = rotated_mesh_x(object_meshes[i], rad)
    object_meshes[i].paint_uniform_color(object_color[:3])

# Save each mesh and store the file path in the vector
for i, mesh in enumerate(mano_meshes):
    file_path = set + "_meshes/" + set + f"_mano_meshes/mano_mesh_{i}.stl" 
    open3d.io.write_triangle_mesh(file_path, mesh)
    file_paths.append(file_path)

for i, mesh in enumerate(object_meshes):
    file_path = set + "_meshes/" + set + f"_object_meshes/object_mesh_{i}.stl"
    open3d.io.write_triangle_mesh(file_path, mesh)
    file_paths.append(file_path)

# Print the file paths
for file_path in file_paths:
    print(file_path)
    
"""
np.savetxt(set + "_meshes/translation_vec.txt", translation_vec)
np.savetxt(set + "_meshes/handJoints3D_vec.txt", handJoints3D_vec)
np.savetxt(set + "_meshes/m0_translation_vec.txt", m0_translation_vec)
np.savetxt(set + "_meshes/realhandtrans_vec.txt", realhandtrans_vec)#
np.savetxt(set + "_meshes/obj_translation_vec.txt", obj_translation_vec)
np.savetxt(set + "_meshes/obj_rotation_vec.txt", obj_rotation_vec)
np.savetxt(set + "_meshes/kypt_inv_trans_vec.txt", kypt_inv_trans_vec)
"""
'''
# Save each mesh and store the file path in the vector
for i, mesh in enumerate(mano_meshes):
    file_path = f"SM1_meshes/SM1_mano_meshes/mano_mesh_{i}.stl" 
    open3d.io.write_triangle_mesh(file_path, mesh)
    file_paths.append(file_path)

for i, mesh in enumerate(object_meshes):
    file_path = f"SM1_meshes/SM1_object_meshes/object_mesh_{i}.stl"
    open3d.io.write_triangle_mesh(file_path, mesh)
    file_paths.append(file_path)

# Print the file paths
for file_path in file_paths:
    print(file_path)

np.savetxt("SM1_meshes/translation_vec.txt", translation_vec)
np.savetxt("SM1_meshes/handJoints3D_vec.txt", handJoints3D_vec)
np.savetxt("SM1_meshes/m0_translation_vec.txt", m0_translation_vec)
'''