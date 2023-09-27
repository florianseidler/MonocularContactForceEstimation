import numpy as np
import torch
from manopth.manolayer import ManoLayer
#from tqdm import tqdm


class PSO:
    def __init__(self, parameters, target, _mano_root='sources/mano/models'):
        """
        particle swarm optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        self.mano_layer = ManoLayer(side="right",
                                    mano_root=_mano_root, use_pca=False, flat_hand_mean=True)

        # 初始化
        self.NGEN = parameters[0]
        self.pop_size = parameters[1]
        self.var_num = parameters[2].shape[1]
        self.bound = []
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
        self.set_target(target)

    def set_target(self, target):
        self.target = target.copy()
        self.pop_x = np.random.randn(self.pop_size, self.var_num)
        self.pop_v = np.random.random((self.pop_size, self.var_num))
        self.p_best = self.pop_x.copy()
        self.p_best_fit = self.batch_new_get_loss(self.pop_x)
        g_best_index = np.argmin(self.p_best_fit, axis=0)
        if g_best_index.shape[0] > 1:
            g_best_index = g_best_index[[0]]
        self.g_best = self.p_best[g_best_index].copy()

    def new_cal_ref_bone(self, _shape):
        parent_index = [0,
                        0, 1, 2,
                        0, 4, 5,
                        0, 7, 8,
                        0, 10, 11,
                        0, 13, 14
                        ]
        index = [0,
                 1, 2, 3,  # index
                 4, 5, 6,  # middle
                 7, 8, 9,  # pinky
                 10, 11, 12,  # ring
                 13, 14, 15]  # thumb
        reoder_index = [
            13, 14, 15,
            1, 2, 3,
            4, 5, 6,
            10, 11, 12,
            7, 8, 9]
        shape = torch.Tensor(_shape.reshape((-1, 10)))
        th_v_shaped = torch.matmul(self.mano_layer.th_shapedirs,
                                   shape.transpose(1, 0)).permute(2, 0, 1) \
                      + self.mano_layer.th_v_template
        th_j = torch.matmul(self.mano_layer.th_J_regressor, th_v_shaped)
        temp1 = th_j.clone().detach()
        temp2 = th_j.clone().detach()[:, parent_index, :]
        result = temp1 - temp2
        result = torch.norm(result, dim=-1, keepdim=True)
        ref_len = result[:, [4]]
        result = result / ref_len
        return torch.squeeze(result, dim=-1)[:, reoder_index].cpu().numpy()

    def batch_new_get_loss(self, beta_):
        weight = 1e-3
        beta = beta_.copy()
        temp = self.new_cal_ref_bone(beta)
        loss = np.linalg.norm(temp - self.target, axis=-1, keepdims=True) ** 2 + \
               weight * np.linalg.norm(beta, axis=-1, keepdims=True)
        return loss

    def update_operator(self, pop_size):

        c1 = 2
        c2 = 2
        w = 0.4

        self.pop_v = w * self.pop_v \
                     + c1 * np.multiply(np.random.rand(pop_size, 1), (self.p_best - self.pop_x)) \
                     + c2 * np.multiply(np.random.rand(pop_size, 1), (self.g_best - self.pop_x))
        self.pop_x = self.pop_x + self.pop_v
        low_flag = self.pop_x < self.bound[0]
        up_flag = self.pop_x > self.bound[1]
        self.pop_x[low_flag] = -3.0
        self.pop_x[up_flag] = 3.0
        temp = self.batch_new_get_loss(self.pop_x)
        p_best_flag = temp < self.p_best_fit
        p_best_flag = p_best_flag.reshape((pop_size,))
        self.p_best[p_best_flag] = self.pop_x[p_best_flag]
        self.p_best_fit[p_best_flag] = temp[p_best_flag]
        g_best_index = np.argmin(self.p_best_fit, axis=0)
        if g_best_index.shape[0] > 1:
            g_best_index = g_best_index[[0]]
        self.g_best = self.pop_x[g_best_index]
        self.g_best_fit = self.p_best_fit[g_best_index][0][0]

    def main(self, slover=None, return_err=False):
        best_fit = []
        self.ng_best = np.zeros((1, self.var_num))
        self.ng_best_fit = self.batch_new_get_loss(self.ng_best)[0][0]
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            dis = self.g_best_fit - self.ng_best_fit
            if self.g_best_fit < self.ng_best_fit:
                self.ng_best = self.g_best.copy()
                self.ng_best_fit = self.g_best_fit
            if abs(dis) < 1e-6:
                break
        if return_err:
            #err = solver.new_cal_ref_bone(self.ng_best)
            #err = align_bone_len(err, self.target)
            return "err"  # err