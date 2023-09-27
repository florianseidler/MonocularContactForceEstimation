import open3d as o3d
import numpy as np
import os

set = "SM1"  # "AP11"  # "SM1"  # "MPM13"  # "MPM14"  # "SB11"  # "GSF10"  # "SMu40"

# Set the initial mesh index
mesh_index = 0


def key_callback_next_frame(vis):
    global mesh_index, hand_meshes, object_meshes, coord_sys_mesh

    mesh_index = (mesh_index + 1) % len(hand_meshes)

    # Update the displayed mesh
    vis.clear_geometries()
    vis.add_geometry(hand_meshes[mesh_index])
    vis.add_geometry(object_meshes[mesh_index])
    vis.add_geometry(coord_sys_mesh)
    vis.update_geometry(hand_meshes[mesh_index])
    vis.update_geometry(object_meshes[mesh_index])
    vis.update_geometry(coord_sys_mesh)
    print("Mesh Number: ", mesh_index)
    vis.poll_events()
    vis.update_renderer()


def key_callback_previous_frame(vis):
    global mesh_index, hand_meshes, object_meshes, coord_sys_mesh

    mesh_index = (mesh_index - 1) % len(hand_meshes)

    # Update the displayed mesh
    vis.clear_geometries()
    vis.add_geometry(hand_meshes[mesh_index])
    vis.add_geometry(object_meshes[mesh_index])
    vis.add_geometry(coord_sys_mesh)
    vis.update_geometry(hand_meshes[mesh_index])
    vis.update_geometry(object_meshes[mesh_index])
    vis.update_geometry(coord_sys_mesh)
    print("Mesh Number: ", mesh_index)
    vis.poll_events()
    vis.update_renderer()


# Function to load meshes from a directory
def load_meshes_from_directory(mesh_dir, color=None):
    meshes = []
    for filename in os.listdir(mesh_dir):
        if filename.endswith(".stl"):
            filepath = os.path.join(mesh_dir, filename)
            mesh = o3d.io.read_triangle_mesh(filepath)
            if color.any():
                mesh.paint_uniform_color(color[:3])
            meshes.append(mesh)
    return meshes


def main():
    # Load the meshes from the directories and add color
    hand_mesh_dir = "output/" + set + "_meshes/" + set + f"_mano_meshes/"
    object_mesh_dir = "output/" + set + "_meshes/" + set + f"_object_meshes/"

    hand_color = np.array([0.960784314, 0.839215686, 0.68627451])
    object_color = np.array([13, 114, 175, 255]) / 255.0  # blue

    hand_meshes = load_meshes_from_directory(hand_mesh_dir, hand_color)
    object_meshes = load_meshes_from_directory(object_mesh_dir, object_color)

    # Load coordinate system for reference
    coord_sys_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Create a visualizer and add initial meshes as geometry
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.add_geometry(coord_sys_mesh)
    vis.add_geometry(hand_meshes[mesh_index])
    vis.add_geometry(object_meshes[mesh_index])

    # Register the key callback function
    vis.register_key_callback(ord("N"), key_callback_next_frame)
    vis.register_key_callback(ord("P"), key_callback_previous_frame)

    # Start the visualization loop
    vis.run()

    # Close the visualizer window
    vis.destroy_window()


if __name__ == "__main__":
    main()
