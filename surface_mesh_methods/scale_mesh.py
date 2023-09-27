import open3d as o3d
import numpy as np


def scaling(mesh, scaling_factor=1.0):
    """
    Scales input mesh by multiplication of vertices and 
    shift center of scaled mesh back towards center of input mesh.

    Return: scaled and translated mesh
    """

    # Get vertices of mesh
    vertices = np.asarray(mesh.vertices)

    # Scale vertices
    scaled_vertices = vertices * scaling_factor

    # Translate vertices
    scaled_and_translated_vertices = translate_vertices(vertices, scaled_vertices)

    scaled_mesh = o3d.geometry.TriangleMesh()
    scaled_mesh.vertices = o3d.utility.Vector3dVector(scaled_and_translated_vertices)
    scaled_mesh.triangles = mesh.triangles
    scaled_mesh.compute_vertex_normals()

    return scaled_mesh


def translate_vertices(vertices, scaled_vertices):
    """
    Return: vertices after subtracting the shift which happend after scaling 
    """
    mean_object = get_center(vertices)
    mean_scaled_object = get_center(scaled_vertices)

    x_offset, y_offset, z_offset = np.subtract(mean_scaled_object, mean_object)

    scaled_vertices[:,0] = np.subtract(scaled_vertices[:,0], x_offset)
    scaled_vertices[:,1] = np.subtract(scaled_vertices[:,1], y_offset)
    scaled_vertices[:,2] = np.subtract(scaled_vertices[:,2], z_offset)
    
    return scaled_vertices


def get_center(matrix):
    """
    Return: center of 3d matrix 
    """
    # get extrema
    min_x = matrix.min(axis=0)[0]
    min_y = matrix.min(axis=0)[1]
    min_z = matrix.min(axis=0)[2]
    max_x = matrix.max(axis=0)[0]
    max_y = matrix.max(axis=0)[1]
    max_z = matrix.max(axis=0)[2]

    x_center = min_x + ((max_x - min_x) / 2.0)
    y_center = min_y + ((max_y - min_y) / 2.0)
    z_center = min_z + ((max_z - min_z) / 2.0)

    return x_center, y_center, z_center    


def create_triangle_mesh(mesh, vertices):
    """
    Return: modified open3d triangular surface mesh with given vertices 
    """
    modified_mesh = o3d.geometry.TriangleMesh()
    modified_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    modified_mesh.triangles = mesh.triangles
    modified_mesh.compute_vertex_normals()

    return modified_mesh
