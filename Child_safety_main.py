# main.py
import cv2
import numpy as np
import time
from ultralytics import YOLO

import config as cfg
import safety_logic as logic


def main():
    # 1. SETUP
    print("Loading YOLO model...")
    model = YOLO(cfg.MODEL_PATH)

    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out = cv2.VideoWriter(cfg.OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Master Memory: { track_id: { ...state data... } }
    track_memory = {}

    # Global Alert Timer for Falls (so message persists)
    global_fall_alert = 0

    print("Starting Pipeline...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()

        # Run YOLO Tracking
        results = model.track(frame, persist=True, verbose=False)

        current_frame_ids = set()
        current_alerts = []

        # Organize detections by class for Interaction Logic
        dets_by_class = {name: [] for name in cfg.CLASS_NAMES}
        doors = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            cls_indices = results[0].boxes.cls.int().cpu().numpy()

            for box, tid, idx in zip(boxes, track_ids, cls_indices):
                label = cfg.CLASS_NAMES[idx]
                center = logic.get_center(box)

                # Store for Interaction Logic
                det_obj = {'box': box, 'id': tid, 'center': center}
                dets_by_class[label].append(det_obj)

                if label == 'open_door':
                    doors.append(box)

                current_frame_ids.add(tid)

                # --- UPDATE TRACK MEMORY ---
                if tid not in track_memory:
                    track_memory[tid] = {
                        'history': [],  # for velocity
                        'lost_count': 0,
                        'alerted_door': False,
                        'class': label
                    }

                # Update basic info
                track_memory[tid]['class'] = label
                track_memory[tid]['lost_count'] = 0  # Reset lost count since seen

                # --- FALL DETECTION LOGIC (Only for Child) ---
                if label == cfg.TARGET_CLASS:
                    # 1. Calculate Velocity
                    velocity = 0
                    prev_y = track_memory[tid]['history'][-1]['y'] if track_memory[tid]['history'] else center[1]
                    velocity = center[1] - prev_y  # Positive = Moving Down

                    # 2. Calculate Pose Angle
                    angle = logic.analyze_pose_for_fall(frame, box)

                    # 3. Store History
                    track_memory[tid]['history'].append(
                        {'y': center[1], 'v': velocity, 'angle': angle, 'time': current_time})
                    if len(track_memory[tid]['history']) > 10:
                        track_memory[tid]['history'].pop(0)

                    # 4. Trigger Fall Alert
                    if (velocity > cfg.VELOCITY_THRESHOLD) or (angle > cfg.FALL_ANGLE_THRESHOLD):
                        global_fall_alert = cfg.FALL_ALERT_DURATION
                        # visual debug on child
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
                        cv2.putText(frame, f"FALL DETECTED! V:{velocity:.1f}", (int(box[0]), int(box[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- INTERACTION LOGIC ---
        for child in dets_by_class[cfg.TARGET_CLASS]:
            # 1. Weapon Touch
            for weapon_label in cfg.DANGER_ITEMS:
                for weapon in dets_by_class[weapon_label]:
                    if logic.is_touching(child['box'], weapon['box']):
                        current_alerts.append(f"SUSPICIOUS: Child touching {weapon_label}!")

            # 2. Hazard Proximity
            for hazard_label in cfg.HAZARDS:
                for hazard in dets_by_class[hazard_label]:
                    dist = np.linalg.norm(np.array(child['center']) - np.array(hazard['center']))
                    if dist < cfg.PROXIMITY_THRESHOLD:
                        current_alerts.append(f"SUSPICIOUS: Child near {hazard_label}")

            # 3. Safe Toy
            for toy in dets_by_class['Toy']:
                if logic.is_touching(child['box'], toy['box']):
                    cv2.putText(frame, "Safe Play", (int(child['box'][0]), int(child['box'][1] - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- DISAPPEARANCE / INVISIBLE LOGIC ---
        # Iterate over known tracks that are NOT in current frame
        for tid in list(track_memory.keys()):
            if tid not in current_frame_ids:
                track_memory[tid]['lost_count'] += 1

                # Get last known data
                history = track_memory[tid]['history']
                if not history: continue
                last_pos = history[-1]
                last_center = (int(track_memory[tid].get('last_center_x', 0)),
                               int(track_memory[tid].get('last_center_y', 0)))  # approximation

                # A. Door Disappearance
                if track_memory[tid]['lost_count'] == cfg.DISAPPEAR_BUFFER_FRAMES:
                    # Check if they vanished inside a door
                    if not track_memory[tid]['alerted_door']:
                        # We use the LAST seen box center from history
                        # (simplified here to use history y, would need x tracking to be perfect,
                        # but relying on `doors` list from THIS frame for checking)
                        pass
                        # Note: In merged logic, checking "last seen inside current door" is tricky
                        # if the door is also not detected. We assume door is static or detected.
                        # For robustness, we check if logic.is_inside(last_known_center, any_current_door)

                        # (Requires storing last center X/Y explicitly in main loop above, added below)

                # B. High Velocity Disappearance (Fall out of frame)
                if track_memory[tid]['class'] == cfg.TARGET_CLASS:
                    last_v = last_pos['v']
                    if last_v > (cfg.VELOCITY_THRESHOLD * 0.8) and track_memory[tid]['lost_count'] < 5:
                        global_fall_alert = cfg.FALL_ALERT_DURATION
                        current_alerts.append("CRITICAL: Child fell out of view!")

                # Cleanup
                if track_memory[tid]['lost_count'] > 30:
                    del track_memory[tid]
            else:
                # Update last known center for next frame's disappearance check
                last_box = dets_by_class[track_memory[tid]['class']][-1]['box']  # simplistic fetch
                # (Ideally, map TID to specific box earlier, but memory dict holds history)

        # --- DRAW VISUALS ---
        annotated_frame = results[0].plot()

        # Draw Global Fall Alert
        if global_fall_alert > 0:
            cv2.putText(annotated_frame, "!!! SUSPICIOUS FALL DETECTED !!!", (50, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
            global_fall_alert -= 1

        # Draw Interaction Alerts
        y_pos = 100
        for msg in set(current_alerts):
            cv2.putText(annotated_frame, msg, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_pos += 40

        out.write(annotated_frame)
        cv2.imshow("Unified Child Safety System", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Saved to {cfg.OUTPUT_PATH}")


if __name__ == "__main__":
    main()