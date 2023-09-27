"""
bash usage: ./FloatTetwild_bin -i input.off -o output.msh
"""
import subprocess
import os 


def apply_ftetwild(base_path, mesh_name):
    current_dir = os.getcwd() 
    abs_dir_path = current_dir + "/" + base_path
    abs_path = abs_dir_path + mesh_name
    abs_stl_path = abs_path + ".stl"

    # Change directory
    os.chdir('fTetWild/build')
    subprocess.run(["./FloatTetwild_bin", "-i", abs_stl_path, "-o", abs_path])
    # Change directory back
    os.chdir('..')
    os.chdir('..')
