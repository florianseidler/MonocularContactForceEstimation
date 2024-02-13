import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from keypoint_transformer.base import Tester
from keypoint_transformer.config import cfg 
from surface_mesh_methods.generate_mano_surf_mesh import get_kypt_mesh
from surface_mesh_methods.object_functions import untransformed_obj_mesh


def kypt_trafo(camera_data=False, object_name=''):

	annotation_available = 0  # 1 for train data

	# setup kypt_transformer
	cfg.use_big_decoder = '--use_big_decoder'
	cfg.dec_layers = 6  # dec_layers
	cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
	cfg.set_args('0', '')  # set gpu_ids

	# choose test mode (train data is with more annotation and other pipeline) 
	tester = Tester('sources/snapshot_21_845.pth.tar')
	tester._make_batch_generator('test', 'all', None, None, None, annotation_available, camera_data, object_name)
	tester._make_model()

	# prepare empty output arrays
	mano_meshes = []
	object_meshes = []
	obj_translation_vec = []
	obj_rotation_vec = []

	joint_vec = []  # for debug

	# iterate through input images listed in HO3D_v3/evaluation.txt
	with torch.no_grad():
		for itr, (inputs, targets, meta_info) in enumerate(tester.batch_generator):
			# forward
			model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)  # model.py forward()
			out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}

			# extract 20 hand joint positions of right hand (without root) 
			# and set root joint of hand to (0,0,0)
			joints_right = out['joint_3d_right'][-1]
			root_joint = torch.zeros((out['joint_3d_right'].shape[1], 1, 3)).to(out['joint_3d_right'].device)
			joints_right = torch.cat([joints_right, root_joint], dim=1)  # .cpu().numpy()
			joints_right = joints_right.cpu().numpy()
			joints_right_3d = joints_right[0]
			
			# create mano surface mesh
			mano_mesh_right_kypt = get_kypt_mesh(joints_right_3d)
			mano_meshes.append(mano_mesh_right_kypt)

			# create untransformed mesh of detected ycb object
			objName = tester.testset.datalist[itr]['objName']
			obj_mesh = untransformed_obj_mesh(objName)
			object_meshes.append(obj_mesh)

			# extract estimated object translation (relative to hand root) 
			kypt_obj_trans = out['obj_trans'][-1]
			kypt_obj_trans = kypt_obj_trans.cpu().numpy()
			obj_translation_vec.append(kypt_obj_trans)

			# extract estimated object rotation
			kypt_obj_rot = out['obj_rot'][-1]
			kypt_obj_rot = kypt_obj_rot.cpu().numpy()
			obj_rotation_vec.append(kypt_obj_rot)

			joint_vec.append(joints_right_3d)

	# Debugging
	from icecream import ic
	def debug_output_to_file(text):
		with open('debug_log.txt', 'a') as f:
			f.write(text + '\n')
	ic.configureOutput(prefix='Debug| ', outputFunction=debug_output_to_file, includeContext=True)
	ic.enable()
	ic(joint_vec)
	ic(obj_rotation_vec)
	ic(obj_translation_vec)

	return (mano_meshes, object_meshes, obj_translation_vec, obj_rotation_vec)


def kypt_trafo_hand_joints():
	"""
	call kypt_transformer to get estimated and annotated poses of the hand from the train dataset
	"""

	# setup kypt_transformer
	cfg.use_big_decoder = '--use_big_decoder'
	cfg.dec_layers = 6  # dec_layers
	cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
	cfg.set_args('0', '')  # set gpu_ids

	# choose test mode (train data is with more annotation and other pipeline) 
	tester = Tester('sources/snapshot_21_845.pth.tar')
	tester._make_batch_generator('test', 'all', None, None, None, annotation_available=1)
	#tester._make_batch_generator('train', 'all', None, None, None, annotation_available=1)
	tester._make_model()

	# prepare empty output arrays
	joints_right_vec = []
	joints_right_3d_vec = []
	joints_right_annotated_vec = []

	# iterate through input images listed in HO3D_v3/annotated.txt
	with torch.no_grad():
		for itr, (inputs, targets, meta_info) in enumerate(tester.batch_generator):
			# forward
			model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)
			out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}

			# extract 20 hand joint positions of right hand (without root) 
			# and set root joint of hand to (0,0,0)
			joints_right = out['joint_3d_right'][-1]
			root_joint = torch.zeros((out['joint_3d_right'].shape[1], 1, 3)).to(out['joint_3d_right'].device)
			joints_right = torch.cat([joints_right, root_joint], dim=1)  # .cpu().numpy()
			joints_right = joints_right.cpu().numpy()
			joints_right_3d = joints_right[0]
			joints_right_annotated = tester.testset.datalist[itr]['handJoints3D']
			
			joints_right_vec.append(joints_right)
			joints_right_3d_vec.append(joints_right_3d)
			joints_right_annotated_vec.append(joints_right_annotated)


	return (joints_right, joints_right_3d, joints_right_annotated_vec)


def kypt_trafo_for_covariance():
	"""
	call kypt_transformer to get estimated and annotated poses of the hand from the train dataset
	"""

	# setup kypt_transformer
	cfg.use_big_decoder = '--use_big_decoder'
	cfg.dec_layers = 6  # dec_layers
	cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
	cfg.set_args('0', '')  # set gpu_ids

	# choose test mode (train data is with more annotation and other pipeline) 
	tester = Tester('sources/snapshot_21_845.pth.tar')
	tester._make_batch_generator('test', 'all', None, None, None, annotation_available=1)
	#tester._make_batch_generator('train', 'all', None, None, None, annotation_available=1)
	tester._make_model()

	# prepare empty output arrays

	obj_rotation_vec = []
	obj_translation_vec = []
	obj_rotation_annotated_vec = []
	obj_translation_annotated_vec = []

	# iterate through input images listed in HO3D_v3/annotated.txt
	with torch.no_grad():
		for itr, (inputs, targets, meta_info) in enumerate(tester.batch_generator):
			# forward
			model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)
			out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}

			# extract estimated object rotation
			kypt_obj_rot = out['obj_rot'][-1]
			kypt_obj_rot = kypt_obj_rot.cpu().numpy()
			obj_rotation_vec.append(kypt_obj_rot)

			# extract estimated object translation (relative to hand root) 
			kypt_obj_trans = out['obj_trans'][-1]
			kypt_obj_trans = kypt_obj_trans.cpu().numpy()

			hand_joint_positions_annotated = tester.testset.datalist[itr]['handJoints3D']
			hand_root_position_annotated = hand_joint_positions_annotated[20] 
			
			obj_translation_vec.append(kypt_obj_trans)
			#obj_translation_vec.append(kypt_obj_trans + hand_root_position_annotated)

			obj_rotation_annotated = tester.testset.datalist[itr]['obj_rot']
			obj_rotation_annotated_vec.append(obj_rotation_annotated)

			obj_translation_annotated = tester.testset.datalist[itr]['obj_trans']
			obj_translation_annotated_vec.append(obj_translation_annotated)


	#return (kypt_obj_rot, kypt_obj_trans, obj_rotation_annotated_vec, obj_translation_annotated_vec)
	return (obj_rotation_vec, obj_translation_vec, obj_rotation_annotated_vec, obj_translation_annotated_vec)


def kypt_trafo_for_covariance_test():
	"""
	call kypt_transformer to get estimated and annotated poses of the hand from the test dataset
	"""

	cfg.use_big_decoder = '--use_big_decoder'
	cfg.dec_layers = 6  
	cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
	cfg.set_args('0', '') 

	tester = Tester('sources/snapshot_21_845.pth.tar')
	tester._make_batch_generator('test', 'all', None, None, None, annotation_available=1)
	tester._make_model()

	# prepare empty output arrays
	obj_translation_vec = []
	obj_translation_annotated_vec = []

	with torch.no_grad():
		for itr, (inputs, targets, meta_info) in enumerate(tester.batch_generator):

			model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)
			out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}
			
			kypt_obj_trans = out['obj_trans'][-1]
			kypt_obj_trans = kypt_obj_trans.cpu().numpy()
			obj_translation_vec.append(kypt_obj_trans)

			obj_translation_annotated = tester.testset.datalist[itr]['obj_trans']
			obj_translation_annotated_vec.append(obj_translation_annotated)

	return obj_translation_vec, obj_translation_annotated_vec