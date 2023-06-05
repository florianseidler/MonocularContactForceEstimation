import cv2
import numpy as np

import mediapipe as mp
from mediapipe.python.solutions.hands import HandLandmark


class MediaPipeHandKeypointEstimator(object):

    def __init__(self):

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.last_world_xyz = [0, 0, 0]
        self.last_joints = np.ones(shape=(16, 3))
        self.last_image = None

        self.mp_hands_estimator = mp.solutions.hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def predict(self, image):
        self.last_image = image

        try:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            img = image

        results = self.mp_hands_estimator.process(img)
        if results.multi_hand_landmarks:
            joints_mp = np.zeros(shape=(21, 3))
            for idx, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                joints_mp[idx] = [landmark.x, landmark.y, landmark.z]

            classification = results.multi_handedness[0].classification
            isLeftHand = True
            if classification[0].label == 'Right':
                isLeftHand = False
            else:
                isLeftHand = True

            #if isLeftHand:
            #    joints_mp = self.flip_left_hand_joints_to_right_hand_joints(joints_mp)

            self.last_world_xyz = self.calc_world_pos_frame_joints_mp(joints_mp)
            self.last_joints = self.mediapipe_to_mano(joints_mp)
            self.last_image = self.get_annotated_landmarks_image(image, results)
            return self.last_world_xyz, self.last_joints, self.last_image
        else:
            return None, None, self.last_image

    def calc_world_pos_frame_joints_mp(self, joints_mp):
        j = joints_mp[HandLandmark.MIDDLE_FINGER_MCP]

        vec_Arr = np.array(joints_mp[HandLandmark.MIDDLE_FINGER_MCP]) - np.array(joints_mp[HandLandmark.WRIST])
        vec_len = np.linalg.norm(vec_Arr)

        world_xyz = [
            j[0] * 0.5,
            j[1] * 0.5,
            1 * vec_len
        ]
        return world_xyz

    def get_connections(self):
        THUMB = ((0, 1), (1, 2), (2, 3), (3, 4))
        INDEX = ((0, 5), (5, 6), (6, 7), (7, 8))
        MIDDLE = ((0, 9), (9, 10), (10, 11), (11, 12))
        RING = ((0, 13), (13, 14), (14, 15), (15, 16))
        LITTLE = ((0, 17), (17, 18), (18, 19), (19, 20))
        HAND_CONNECTIONS = frozenset().union(*[
            THUMB, INDEX, MIDDLE, RING, LITTLE
        ])
        return HAND_CONNECTIONS

    def mediapipe_to_mano(self, joints3d):
        joints = np.zeros(shape=(21, 3))
        joints[0] = joints3d[HandLandmark.WRIST]
        joints[1] = joints3d[HandLandmark.INDEX_FINGER_MCP]
        joints[2] = joints3d[HandLandmark.INDEX_FINGER_PIP]
        joints[3] = joints3d[HandLandmark.INDEX_FINGER_DIP]
        joints[4] = joints3d[HandLandmark.MIDDLE_FINGER_MCP]
        joints[5] = joints3d[HandLandmark.MIDDLE_FINGER_PIP]
        joints[6] = joints3d[HandLandmark.MIDDLE_FINGER_DIP]
        joints[7] = joints3d[HandLandmark.PINKY_MCP]
        joints[8] = joints3d[HandLandmark.PINKY_PIP]
        joints[9] = joints3d[HandLandmark.PINKY_DIP]
        joints[10] = joints3d[HandLandmark.RING_FINGER_MCP]
        joints[11] = joints3d[HandLandmark.RING_FINGER_PIP]
        joints[12] = joints3d[HandLandmark.RING_FINGER_DIP]
        joints[13] = joints3d[HandLandmark.THUMB_CMC]
        joints[14] = joints3d[HandLandmark.THUMB_MCP]
        joints[15] = joints3d[HandLandmark.THUMB_IP]
        joints[16] = joints3d[HandLandmark.INDEX_FINGER_TIP]
        joints[17] = joints3d[HandLandmark.MIDDLE_FINGER_TIP]
        joints[18] = joints3d[HandLandmark.PINKY_TIP]
        joints[19] = joints3d[HandLandmark.RING_FINGER_TIP]
        joints[20] = joints3d[HandLandmark.THUMB_TIP]
        joints = joints - joints3d[HandLandmark.MIDDLE_FINGER_MCP]
        return joints

    def get_annotated_landmarks_image(self, image, results):
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
        return annotated_image

    @staticmethod
    def flip_left_hand_joints_to_right_hand_joints(left_hand_joints):
        right_hand_joints = np.zeros(shape=(21, 3))
        for i in range(21):
            right_hand_joints[i] = left_hand_joints[i]
            right_hand_joints[i][0] = left_hand_joints[i][0] * -1
            right_hand_joints[i][2] = left_hand_joints[i][2] * -1
            right_hand_joints[i][1] = left_hand_joints[i][1] * -1

        return right_hand_joints
