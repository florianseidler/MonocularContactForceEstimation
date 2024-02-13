import numpy as np
import os
import pytransform3d.visualizer as pv
from distance3d import visualization, hydroelastic_contact
from grasp_metrics.grasp_configuration import Contacts
from contact_force_estimation.contact_functions import (create_rigid_body, create_point_contact, 
                                                        create_surface_contact, print_metrics_result, 
                                                        print_geo_metrics_results)  # print_area_of_grasp_polygon
from icecream import ic

"""
adjustable parameters
---------------------
"""
friction_coef = 0.6  # friction between hand and object  1.0
hand_youngs_modulus = 0.5  # elasticity module of hand  11.0
object_youngs_modulus = 2400.0  # elasticity module of object plastic: 2400, metal: 200000, 1.0 

set = "DSCtest"  # "AP11"  # "SM1"  # "MPM13"  # "MPM14"  # "SB11"  # "GSF10"  # "SMu40"  # "DSCtest"
idx = "6"

scaled = 0


def main():
    curr_dir = os.getcwd()
    base_dir = curr_dir + "/output/" + set + "_meshes/"

    tetrahedral_hand_mesh = ic(base_dir + set + "_mano_meshes/mano_mesh_" + idx + "_.vtk")
    hand_potentials = ic(base_dir + set + "_mano_meshes/mano_potentials_" + idx + ".json")

    if scaled:
        tetrahedral_object_mesh = ic(base_dir + set + "_object_meshes/object_mesh_scaled_" + idx + "_.vtk")
        object_potentials = ic(base_dir + set + "_object_meshes/object_potentials_scaled_" + idx + ".json")
    else:
        tetrahedral_object_mesh = ic(base_dir + set + "_object_meshes/object_mesh_" + idx + "_.vtk")
        object_potentials = ic(base_dir + set + "_object_meshes/object_potentials_" + idx + ".json")

    # load rigid bodies
    hand = create_rigid_body(tetrahedral_hand_mesh, hand_potentials)
    object = create_rigid_body(tetrahedral_object_mesh, object_potentials)

    # adjust elasticity
    hand.youngs_modulus = hand_youngs_modulus
    object.youngs_modulus = object_youngs_modulus

    # apply hydroelastic contact force model
    intersection, wrench_hand_object, wrench_object_hand, details_hand_object = (
        hydroelastic_contact.contact_forces(hand, object, return_details=True))
    assert intersection

    # create point contacts
    point_contacts = Contacts([
        create_point_contact(wrench_object_hand, details_hand_object, friction_coef),
    ])

    print("\nPoint contact model:\n")
    print_metrics_result(point_contacts)

    # create surface contacts
    surface_contacts = Contacts(
        create_surface_contact(details_hand_object, friction_coef)
    )

    print("\n\nSurface contact model:\n")
    print_metrics_result(surface_contacts)
    
    if scaled:
        print_geo_metrics_results(details_hand_object["contact_coms"], base_dir + set + "_object_meshes/object_mesh_scaled_" + idx + ".stl")
    else:
        print_geo_metrics_results(details_hand_object["contact_coms"], base_dir + set + "_object_meshes/object_mesh_" + idx + ".stl")

    # visualize rigid bodies and contact surface(s)
    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.1)
    visualization.RigidBodyTetrahedralMesh(
        hand.body2origin_, hand.vertices_, hand.tetrahedra_).add_artist(fig)
    visualization.RigidBodyTetrahedralMesh(
        object.body2origin_, object.vertices_, object.tetrahedra_).add_artist(fig)

    contact_surface_hand_sphere = visualization.ContactSurface(
        np.eye(4),
        details_hand_object["contact_polygons"],
        details_hand_object["contact_polygon_triangles"],
        details_hand_object["pressures"])
    contact_surface_hand_sphere.add_artist(fig)
    
    fig.view_init()
    fig.show()


if __name__ == "__main__":
    main()
