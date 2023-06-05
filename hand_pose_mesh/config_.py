

#use for paz_minimal_hand_estimator/main.py demo and paz_minimal_hand_estimator/prepare_mano.py:
#MANO_CONF_ROOT_DIR = './'
#use for human_hand_rgb_to_robot/main.py:
MANO_CONF_ROOT_DIR = "paz_minimal_hand_estimator/"


# Attetion: pretrained detnet and iknet are only trained for left model! use left hand
DETECTION_MODEL_PATH = MANO_CONF_ROOT_DIR+'model/detnet/detnet.ckpt'
IK_MODEL_PATH = MANO_CONF_ROOT_DIR+'model/iknet/iknet.ckpt'

# Convert 'HAND_MESH_MODEL_PATH' to 'HAND_MESH_MODEL_PATH_JSON' with 'prepare_mano.py'
print(MANO_CONF_ROOT_DIR)
HAND_MESH_MODEL_LEFT_PATH_JSON = MANO_CONF_ROOT_DIR+'model/hand_mesh/mano_hand_mesh_left.json'
HAND_MESH_MODEL_RIGHT_PATH_JSON = MANO_CONF_ROOT_DIR+'model/hand_mesh/mano_hand_mesh_right.json'

OFFICIAL_MANO_LEFT_PATH = MANO_CONF_ROOT_DIR+'model/mano/MANO_LEFT.pkl'
OFFICIAL_MANO_RIGHT_PATH = MANO_CONF_ROOT_DIR+'model/mano/MANO_RIGHT.pkl'

IK_UNIT_LENGTH = 0.09473151311686484 # in meter

#visualization (replaceable with hand_mesh_model.json?)
OFFICIAL_MANO_PATH_LEFT_JSON = MANO_CONF_ROOT_DIR+'model/mano_handstate/mano_left.json'
OFFICIAL_MANO_PATH_RIGHT_JSON = MANO_CONF_ROOT_DIR+'model/mano_handstate/mano_right.json'

#HAND_COLOR = [228/255, 178/255, 148/255]

# only for rendering
#CAM_FX = 620.744
#CAM_FY = 621.151
