import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from common.base import Tester
from common.config import cfg 
import numpy as np


def kypt_trafo_old():

    cfg.use_big_decoder = '--use_big_decoder'
    cfg.dec_layers = 6  # dec_layers
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    cfg.set_args('0', '')  # set gpu_ids

    tester = Tester('sources/snapshot_21_845.pth.tar')
    tester._make_batch_generator('test', 'all', None, None, None)
    tester._make_model()

    joints_right_vec = []
    translation_vec = []
    handJoints3D_vec = []
    mano_meshes = []
    object_meshes = []
    m0_translation_vec = []
    realhandtrans_vec = []
    obj_translation_vec = []
    obj_rotation_vec = []
    kypt_inv_trans_vec = []

    from hand_pose_mesh.mano import HandState
    hand_state_left = HandState(left=True)
    hand_state_right = HandState(left=False)
    mano_2_world_transform = np.eye(4)


    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tester.batch_generator):
            #print("itr:", itr)
            # forward
            model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)
            out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}

            root_joint = torch.zeros((out['joint_3d_right'].shape[1], 1, 3)).to(out['joint_3d_right'].device)

            joints_right = out['joint_3d_right'][-1]
            joints_right = torch.cat([joints_right, root_joint], dim=1)  # .cpu().numpy()
            joints_right = joints_right.cpu().numpy()
            joints_3d = joints_right[0]
            joints_right_vec.append(joints_right[0])

            kypt_obj_rot = out['obj_rot'][-1]
            kypt_obj_rot = kypt_obj_rot.cpu().numpy()
            kypt_obj_trans = out['obj_trans'][-1]
            kypt_obj_trans = kypt_obj_trans.cpu().numpy()

            kypt_inv_trans = out['inv_trans']
            kypt_inv_trans = np.squeeze(kypt_inv_trans.cpu().numpy())
            kypt_inv_trans_vec = np.append(kypt_inv_trans[0], kypt_inv_trans[1])
            
            translation = out['rel_trans'].cpu().numpy()
            translation_vec.append(translation[0])

            realhandtrans = tester.testset.datalist[itr]['realhandtrans']
            realhandtrans_vec.append(realhandtrans)
            
            #objLabel = tester.testset.datalist[itr].objLabel
            handJoints3D = tester.testset.datalist[itr]['handJoints3D']
            handJoints3D_vec.append(handJoints3D)

            from hand_pose_mesh.MANO_pose_generator import get_kypt_mesh
            mano_mesh_kypt, m0_translation, kypt_handstate = get_kypt_mesh(joints_3d)
            m0_translation_vec.append(m0_translation)
            #mano_mesh_kypt.translate(handJoints3D)
            mano_meshes.append(mano_mesh_kypt)

            objName = tester.testset.datalist[itr]['objName']
            obj_trans = tester.testset.datalist[itr]['objTranslation']  # out['obj_trans'].cpu().numpy()
            obj_trans = kypt_obj_trans  # from kypt trafo
            obj_rot = tester.testset.datalist[itr]['objRotation']  # out['obj_rot'].cpu().numpy()
            obj_rot = np.squeeze(obj_rot)
            obj_rot = kypt_obj_rot  # from kypt trafo
            obj_translation_vec.append(obj_trans)
            obj_rotation_vec.append(obj_rot)
            #obj_trans = [0, 0, 0]
            #hand_state_left = kypt_handstate
            #hand_state_right = kypt_handstate
            #from methods import get_object_mesh
            #object_meshes.append(get_object_mesh(obj_trans, obj_rot, objName, handJoints3D, hand_state_left, hand_state_right, mano_2_world_transform))
            #objMesh.rotate(objMesh.get_rotation_matrix_from_xyz(kypt_obj_rot[3:]), center=(0, 0, 0))
            #object_meshes.append(objMesh)
            from methods import pure_obj_mesh
            obj_mesh = pure_obj_mesh(objName)
            object_meshes.append(obj_mesh)
            

    return (mano_meshes, object_meshes, translation_vec, 
            handJoints3D_vec, m0_translation_vec, realhandtrans_vec, 
            obj_translation_vec, obj_rotation_vec,
            kypt_inv_trans_vec)  # joints_right_vec, translation_vec


def kypt_trafo():

    cfg.use_big_decoder = '--use_big_decoder'
    cfg.dec_layers = 6  # dec_layers
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    cfg.set_args('0', '')  # set gpu_ids

    tester = Tester('sources/snapshot_21_845.pth.tar')
    tester._make_batch_generator('test', 'all', None, None, None)
    tester._make_model()

    #mano_meshes = []
    #object_meshes = []
    
    joint_matrices = []
    obj_translation_vec = []
    obj_rotation_vec = []


    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tester.batch_generator):

            model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)
            out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}

            # create hand mesh from 3d keypoints of transformer
            root_joint = torch.zeros((out['joint_3d_right'].shape[1], 1, 3)).to(out['joint_3d_right'].device)
            joints_right = out['joint_3d_right'][-1]
            joints_right = torch.cat([joints_right, root_joint], dim=1)
            joints_right = joints_right.cpu().numpy()
            joints_3d = joints_right[0]
            #from hand_pose_mesh.MANO_pose_generator import get_kypt_mesh
            #mano_mesh_kypt, m0_translation, kypt_handstate = get_kypt_mesh(joints_3d)
            #mano_meshes.append(mano_mesh_kypt)
            
            # load object mesh
            objName = tester.testset.datalist[itr]['objName']
            #from methods import pure_obj_mesh
            #raw_obj_mesh = pure_obj_mesh(objName)

            # load estimated rotation of the object
            kypt_obj_rot = out['obj_rot'][-1]
            obj_rot = kypt_obj_rot.cpu().numpy()
            
            # load estimated translation of the object
            kypt_obj_trans = out['obj_trans'][-1]
            obj_trans  = kypt_obj_trans.cpu().numpy()

            #from object_functions import kypt_demo_obj_mesh
            obj_rot = np.array(obj_rot)
            obj_trans = np.array(obj_trans)
            #obj_mesh_kypt = kypt_demo_obj_mesh(raw_obj_mesh, obj_rot, obj_trans)
            #object_meshes.append(obj_mesh_kypt)
            
            joint_matrices.append(joints_3d)
            obj_rotation_vec.append(obj_rot)
            obj_translation_vec.append(obj_trans)

    return joint_matrices, obj_rotation_vec, obj_translation_vec, objName  # mano_meshes, object_meshes