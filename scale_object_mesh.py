import open3d as o3d
from surface_mesh_methods.scale_mesh import scaling
#import numpy as np


set = 'SM1'  # "AP11"  # "SM1"  # "MPM13"  # "MPM14"  # "SB11"  # "GSF10"  # "SMu40"  # "DSCtest"
idx = "0"
file_path = "output/" + set + "_meshes/" + set + f"_object_meshes/object_mesh_" + idx + ".stl" 
file_path_scaled = "output/" + set + "_meshes/" + set + f"_object_meshes/object_mesh_scaled_" + idx + ".stl" 


def main():
    # Load STL file
    object = o3d.io.read_triangle_mesh(file_path)

    dilated_object = scaling(object, scaling_factor=1.35)

    # Visualize the dilated mesh
    o3d.visualization.draw_geometries([object, dilated_object])
    
    # Save the dilated mesh
    o3d.io.write_triangle_mesh(file_path_scaled, dilated_object)


if __name__ == "__main__":
    main()
