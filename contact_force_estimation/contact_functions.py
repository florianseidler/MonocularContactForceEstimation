import numpy as np
import json
from distance3d import hydroelastic_contact, utils, io
from grasp_metrics.grasp_configuration import Contact
from grasp_metrics.quasi_static_point_based import (ferrari_canny, force_closure, wrench_resistance, 
                                                    grasp_isotropy, min_singular, wrench_volume)


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


def print_metrics_result(contacts):
    
    print(f"{force_closure(contacts)=}")
    print(f"{ferrari_canny(contacts)=}")
    print(f"{grasp_isotropy(contacts)=}")
    print(f"{min_singular(contacts)=}")
    print(f"{wrench_volume(contacts)=}")
    print(f"{wrench_resistance(contacts, target_wrench=np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))=}")
    print(f"{wrench_resistance(contacts, target_wrench=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))=}")

