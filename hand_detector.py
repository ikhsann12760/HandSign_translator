import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

class HandDetector:
    def __init__(self, model_path='hand_landmarker.task', num_hands=2, min_detection_confidence=0.5, min_presence_confidence=0.5, min_tracking_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, image):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect landmarks
        self.results = self.detector.detect(mp_image)
        return self.results

    def draw_landmarks(self, image, hand_landmarks, mirrored=False):
        # MediaPipe's drawing_utils is not directly compatible with Tasks API landmarks
        # So we draw manually
        h, w, c = image.shape
        
        # Connections for hand landmarks
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8), # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17) # Palm
        ]

        # Convert landmarks to pixel coordinates
        pixel_landmarks = []
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            if mirrored:
                cx = w - cx
            pixel_landmarks.append((cx, cy))

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            cv2.line(image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)

        # Draw joints
        for cx, cy in pixel_landmarks:
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

def calc_landmark_list(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []

    # Keypoint
    for landmark in hand_landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([landmark_x, landmark_y])

    return landmark_list

def pre_process_landmark(landmark_list):
    temp_landmark_list = landmark_list.copy()

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    import itertools
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
