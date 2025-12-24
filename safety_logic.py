# safety_logic.py
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose once
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)


def get_center(box):
    """Returns center (x, y) of a bounding box."""
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))


def is_inside(point, box):
    """Checks if a point is inside a box [x1, y1, x2, y2]."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def is_touching(box_a, box_b):
    """Checks if two boxes overlap."""
    return not (box_a[2] < box_b[0] or box_a[0] > box_b[2] or box_a[3] < box_b[1] or box_a[1] > box_b[3])


def calculate_angle(p1, p2):
    """Calculates angle between two points relative to vertical."""
    return np.degrees(np.arctan2(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])))


def analyze_pose_for_fall(frame, box):
    """
    Extracts ROI and calculates body angle using MediaPipe.
    Returns: angle (float)
    """
    x1, y1, x2, y2 = map(int, box)
    # Clamp coordinates to frame dimensions
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0

    results = pose_estimator.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    angle = 0

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # Calculate angle using Mid-Shoulder and Mid-Hip
        # Note: landmarks are normalized [0,1], so we use them directly for atan2
        s_mid = [(lm[11].x + lm[12].x) / 2, (lm[11].y + lm[12].y) / 2]
        h_mid = [(lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2]
        angle = calculate_angle(s_mid, h_mid)

    return angle