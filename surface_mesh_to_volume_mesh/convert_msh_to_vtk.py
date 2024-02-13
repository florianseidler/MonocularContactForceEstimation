"""
# usage: convert_msh_to_vtk(file_path)

other option:

import subprocess
subprocess.run(["gmsh", "-format", "vtk", "-save", "-o", "mano.vtk", "mano_.msh"])
"""

import gmsh


def convert_msh_to_vtk(file_path):
    """
    Uses gmsh to open a msh file, converts it and 
    saves it as a vtk file in the same direction.
    """

    gmsh.initialize() 
    input_file_path = file_path + "_.msh"
    
    # load msh file
    gmsh.open(input_file_path) 

    # save vtk file
    output_file_path = file_path + "_.vtk"
    gmsh.write(output_file_path) 
    gmsh.finalize() 


def convert_msh_to_vtk_todel(file_path):
    # TODO: delete function
    """
    Uses gmsh to open a msh file, converts it and 
    saves it as a vtk file in the same direction.
    """

    gmsh.initialize() 

    input_file_path = file_path + "_.msh"

    gmsh.open(input_file_path) # load msh file  #base_file_path
    """
    import re
    pattern = r"msh$" 
    replacement = "vtk" 
    # replace pattern with replacement
    
    output_file_path = re.sub(pattern, replacement, input_file_path)
    """
    output_file_path = file_path + "_.vtk"

    #import os 
    #curr_dir = os.getcwd()

    gmsh.write(output_file_path) # save vtk file
    gmsh.finalize() 

