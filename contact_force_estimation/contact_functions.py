import numpy as np
import json
from distance3d import hydroelastic_contact, utils, io
from grasp_metrics.grasp_configuration import Contact
from grasp_metrics.quasi_static_point_based import (ferrari_canny, force_closure, wrench_resistance, 
                                                    grasp_isotropy, min_singular, wrench_volume)
from scipy.spatial import ConvexHull
import trimesh


def create_rigid_body(tetrahedral_mesh, potentials):
    vertices, tetrahedra = io.load_tetrahedral_mesh(tetrahedral_mesh)
    with open(potentials, "r") as f:
        potentials = np.asarray(json.load(f))
    return hydroelastic_contact.RigidBody(np.eye(4), vertices, tetrahedra, potentials)


def create_point_contact(wrench, details, friction_coef=100.0):
    normal = utils.norm_vector(wrench[:3])
    force = np.linalg.norm(wrench[:3])
    return Contact(details["contact_point"], normal, force, friction_coef)


def create_surface_contact(details, friction_coef=1.0):
    contact_points = details["contact_coms"]
    contact_force_vectors = details["contact_forces"]
    contact_forces = np.linalg.norm(contact_force_vectors, axis=1)
    denom = np.copy(contact_forces)
    denom[denom < np.finfo("float").eps] = 1.0
    contact_normals = contact_force_vectors / denom[:, np.newaxis]
    return [
        Contact(contact_points[i], contact_normals[i], contact_forces[i], friction_coef)
        for i in range(len(contact_points))
    ]


def volume_of_grasp_polygon(contact_points, surf_object_mesh_dir):
    """Area of the grasp polygon defined by 3 or more fingers.

    Parameters
    ----------
    contact_points : array-like, shape (n_contacts, 3)
        Contact points.

    surf_object_mesh_dir : directory
        Directory of the object surface mesh STL-file.

    Returns
    -------
    covered_vol : float
        Metric based on volume of the grasp polygon. covered_vol 
        shows how much of the object's volume is covered by the grasp. 
        Larger values are better.
        covered_vol = 0: no grasp volume -> less than 2 contact points
        covered_vol < 1: volume is partly covered
        covered_vol = 1: whole volume is covered
    """

    # volume of the grasp polygon via convex hull
    grasp_hull = ConvexHull(contact_points)
    grasp_volume = grasp_hull.volume

    # volume of the object polygon via surface mesh
    object_surf_mesh = trimesh.load(surf_object_mesh_dir)
    object_volume = object_surf_mesh.volume

    # share of the grasp volume
    covered_vol = grasp_volume / object_volume

    return covered_vol


def print_metrics_result(contacts):
    
    print(f"{force_closure(contacts)=}")
    print(f"{ferrari_canny(contacts)=}")
    print(f"{grasp_isotropy(contacts)=}")
    print(f"{min_singular(contacts)=}")
    print(f"{wrench_volume(contacts)=}")
    print(f"{wrench_resistance(contacts, target_wrench=np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))=}")
    print(f"{wrench_resistance(contacts, target_wrench=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))=}")
    
    
def print_geo_metrics_results(contact_points, surf_object_mesh_dir):
    print("volume of grasp polygon:", volume_of_grasp_polygon(contact_points, surf_object_mesh_dir))
