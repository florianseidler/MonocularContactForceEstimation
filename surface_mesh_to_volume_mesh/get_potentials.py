import os
import subprocess
import re


def apply_compute_potentials(file_path, mesh_name):
    """
    Invoces compute_potentials.py from distance3d via bash 
    """
    stl_file_path = file_path + mesh_name + ".stl"
    vtk_file_path = file_path + mesh_name + "_.vtk"

    # replace "mesh" with "potentials"
    pattern = r"mesh" 
    replacement = "potentials" 
    output_potential_name = re.sub(pattern, replacement, mesh_name)

    output_file_path = file_path + output_potential_name + ".json"  

    current_dir = os.getcwd() 
    output_file_path = current_dir + "/" + output_file_path
    
    subprocess.run(["python3", "distance3d/bin/compute_potentials.py", stl_file_path, vtk_file_path, output_file_path])