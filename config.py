
# --- FILE PATHS ---
MODEL_PATH = "../new_best.pt"
VIDEO_PATH = "../TV/11. Fall Abnormal.mp4" # Update this to your target video
OUTPUT_PATH = "2._safety_output.mp4"

# --- CLASS DEFINITIONS ---
# Ensure these match your YOLO model's training exactly
CLASS_NAMES = ['Adult', 'Child', 'Fire', 'Stairs', 'Toy', 'knife', 'open_door', 'scissors']

# subsets for logic
DANGER_ITEMS = ['knife', 'scissors']
HAZARDS = ['Fire', 'Stairs']
TARGET_CLASS = 'Child' # The class we monitor for falls (was 'baby' in script 2)

# --- INTERACTION THRESHOLDS ---
PROXIMITY_THRESHOLD = 100 # Pixel distance for hazards
DISAPPEAR_BUFFER_FRAMES = 5 # Frames to wait before confirming disappearance

# --- FALL DETECTION THRESHOLDS ---
FALL_ANGLE_THRESHOLD = 60    # Degrees (Horizontal lean)
VELOCITY_THRESHOLD = 15.0    # Pixels per frame (Downward speed)
MISSING_FRAME_TOLERANCE = 10
FALL_ALERT_DURATION = 20     # How long the alert stays on screen