import numpy as np
import torch

from .particleSwarmOptimization import PSO
from .LM_solver import LM_Solver, caculate_length


def pso_betas(keypoints):
    solver = LM_Solver(num_Iter=500, th_beta=torch.zeros((1, 10)), th_pose=torch.zeros((1, 48)),
                        lb_target=np.zeros((15, 1)),
                        weight=1e-5)

    NGEN = 100
    popsize = 100
    low = np.zeros((1, 10)) - 3.0
    up = np.zeros((1, 10)) + 3.0
    parameters = [NGEN, popsize, low, up]
    keypoints = caculate_length(keypoints, label='useful')
    keypoints = keypoints.reshape((1, 15))
    pso = PSO(parameters, keypoints)
    pso.main(slover=solver, return_err=False)
    return pso.ng_best
