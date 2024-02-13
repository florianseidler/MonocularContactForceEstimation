from surface_mesh_to_volume_mesh.get_msh_mesh import apply_ftetwild
from surface_mesh_to_volume_mesh.convert_msh_to_vtk import convert_msh_to_vtk
from surface_mesh_to_volume_mesh.get_potentials import apply_compute_potentials


set = 'SM1'  # "AP11"  # "SM1"  # "MPM13"  # "MPM14"  # "SB11"  # "GSF10"  # "SMu40"  # "DSCtest"
idx = "0"
meshtype = "mano"   # "mano" (hand) or "object"
#meshtype = "object"   # "mano" (hand) or "object"
scaled = 0

def main():

    base_path = "output/" + set + "_meshes/" + set + f"_" + meshtype + "_meshes/" 
    if scaled:
        mesh_name = meshtype + "_mesh_scaled_" + idx 
    else:
        mesh_name = meshtype + "_mesh_" + idx 
    base_file_path = base_path + mesh_name

    apply_ftetwild(base_path, mesh_name)
    convert_msh_to_vtk(base_file_path)
    apply_compute_potentials(base_path, mesh_name)


if __name__ == "__main__":
    main()
