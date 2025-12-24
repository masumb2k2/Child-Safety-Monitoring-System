import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp

# --- CONFIGURATION ---
MODEL_PATH = "../new_best.pt"
VIDEO_PATH = "../VDo/11. Fall Abnormal.mp4"

# Class Mapping (Based on your provided list)
CLASS_NAMES = ['Adult', 'Child', 'Fire', 'Stairs', 'Toy', 'knife', 'open_door', 'scissors']
DANGER_ITEMS = ['knife', 'scissors']
HAZARDS = ['Fire', 'Stairs']

# Thresholds
FALL_ANGLE_THRESHOLD = 60  # Degrees
VELOCITY_THRESHOLD = 15.0  # Pixels per frame
PROXIMITY_THRESHOLD = 100  # Pixels
DISAPPEAR_BUFFER = 5  # Frames to confirm door crossing

# Initialize Models
model = YOLO(MODEL_PATH)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize Video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter("integrated_safety_pipeline.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Persistent Memory: {id: {'last_y': 0, 'lost_count': 0, 'class': '', 'last_center': (0,0)}}
track_memory = {}


def get_angle(lm):
    """Calculates body angle from pose landmarks."""
    s_mid = [(lm[11].x + lm[12].x) / 2, (lm[11].y + lm[12].y) / 2]
    h_mid = [(lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2]
    return np.degrees(np.arctan2(abs(h_mid[0] - s_mid[0]), abs(h_mid[1] - s_mid[1])))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, verbose=False)
    current_alerts = []
    current_frame_ids = set()

    # Storage for spatial logic
    dets = {name: [] for name in CLASS_NAMES}

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        cls_indices = results[0].boxes.cls.int().cpu().numpy()

        for box, tid, idx in zip(boxes, track_ids, cls_indices):
            label = CLASS_NAMES[idx]
            x1, y1, x2, y2 = map(int, box)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            current_frame_ids.add(tid)

            # 1. ORGANIZE DETECTIONS
            dets[label].append({'box': box, 'id': tid, 'center': center})

            # 2. UPDATE MEMORY & FALL LOGIC (Requirement 1)
            if label in ['Child', 'Adult']:  # Monitoring both for specific logic
                # Calculate velocity
                velocity = 0
                if tid in track_memory:
                    velocity = center[1] - track_memory[tid]['last_center'][1]

                # Pose Analysis for Fall
                roi = frame[max(0, y1 - 5):min(height, y2 + 5), max(0, x1 - 5):min(width, x2 + 5)]
                angle = 0
                if roi.size > 0:
                    res = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    if res.pose_landmarks:
                        angle = get_angle(res.pose_landmarks.landmark)

                # Fall Trigger
                if (velocity > VELOCITY_THRESHOLD or angle > FALL_ANGLE_THRESHOLD) and label == 'Child':
                    current_alerts.append(f"SUSPICIOUS: Fall Detected for Child {tid}!")

                # Store State
                track_memory[tid] = {'last_center': center, 'class': label, 'lost_count': 0}

    # 3. INTERACTION & PROXIMITY LOGIC (Requirements 2 & 3)
    for child in dets['Child']:
        # Dangerous Items (Knife/Scissors)
        for d_label in DANGER_ITEMS:
            for item in dets[d_label]:
                if not (child['box'][2] < item['box'][0] or child['box'][0] > item['box'][2] or
                        child['box'][3] < item['box'][1] or child['box'][1] > item['box'][3]):
                    current_alerts.append(f"CRITICAL: Child {child['id']} touching {d_label}!")

        # Hazards (Fire/Stairs)
        for h_label in HAZARDS:
            for hazard in dets[h_label]:
                dist = np.linalg.norm(np.array(child['center']) - np.array(hazard['center']))
                if dist < PROXIMITY_THRESHOLD:
                    current_alerts.append(f"WARNING: Child {child['id']} near {h_label}!")

    # 4. DOOR CROSSING / VANISHING LOGIC (Requirement 4)
    for tid in list(track_memory.keys()):
        if tid not in current_frame_ids:
            track_memory[tid]['lost_count'] += 1

            # If they just disappeared, check if it was near an open door
            if track_memory[tid]['lost_count'] == DISAPPEAR_BUFFER:
                last_center = track_memory[tid]['last_center']
                subject_type = track_memory[tid]['class']

                for door in dets['open_door']:
                    d_box = door['box']
                    if d_box[0] <= last_center[0] <= d_box[2] and d_box[1] <= last_center[1] <= d_box[3]:
                        current_alerts.append(f"SUSPICIOUS: {subject_type} {tid} crossed Open Door and vanished!")

            if track_memory[tid]['lost_count'] > 30:
                del track_memory[tid]

    # --- VISUALIZATION ---
    annotated_frame = results[0].plot()
    y_offset = 50
    for alert in set(current_alerts):
        cv2.putText(annotated_frame, alert, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 40

    out.write(annotated_frame)
    cv2.imshow("Integrated Child Safety Pipeline", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()