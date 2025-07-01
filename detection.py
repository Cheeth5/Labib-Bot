import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import time

pygame.init()

# Initialize models
object_model = YOLO("/home/falku/Desktop/exhibition/yolov8n.pt").to('cuda')  # Object detection
pose_model = YOLO("yolov8n-pose.pt").to('cuda')  # Pose estimation

# Open camera
cap = cv2.VideoCapture(0)

# Bottle tracking variables
previous_bottle_center = None
last_bottle_time = None
bottle_missing_duration = 0
bottle_missing_threshold = 1.5  # seconds

# Alert variables
throwing_detected = False
flash_duration = 30
flash_counter = 0
sound_played = False
first_sound_finished = False

# Load resources
alert_img = cv2.imread("alert.jpg")
alert_img = cv2.resize(alert_img, (640, 480))
sound1 = pygame.mixer.Sound('/home/falku/Desktop/exhibition/0.mp3')
sound2 = pygame.mixer.Sound('/home/falku/Desktop/exhibition/voice.mp3')
sound1_duration = sound1.get_length()
sound2_duration = sound2.get_length()

# Display setup
cv2.namedWindow("YOLO Pose Detection", cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty("YOLO Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# FPS counter
prev_time = time.time()

# Keypoint connections for drawing skeleton
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head to shoulders to elbows
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Shoulders to wrists
    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Hips to ankles
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        break
    
    frame_resized = cv2.resize(frame, (640, 480))
    current_time = time.time()
    
    # Calculate FPS
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Object Detection
    bottle_center = None
    object_results = object_model(frame_resized, half=True)
    for result in object_results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            if int(class_id) == 39:  # Bottle class
                bottle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_resized, "Bottle", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Pose Detection with YOLOv8
    pose_results = pose_model(frame_resized, half=True, device=0)
    for result in pose_results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy[0].cpu().numpy()
            
            # Draw keypoints
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame_resized, (x, y), 5, (255, 0, 0), -1)
            
            # Draw skeleton
            for i, j in skeleton:
                if i < len(keypoints) and j < len(keypoints):
                    start = (int(keypoints[i][0]), int(keypoints[i][1]))
                    end = (int(keypoints[j][0]), int(keypoints[j][1]))
                    cv2.line(frame_resized, start, end, (255, 0, 0), 2)

    # Improved throwing detection logic
    if bottle_center is not None:
        previous_bottle_center = bottle_center
        last_bottle_time = current_time
        bottle_missing_duration = 0
        throwing_detected = False  # Reset if bottle reappears
    elif last_bottle_time is not None:
        bottle_missing_duration = current_time - last_bottle_time
        if bottle_missing_duration >= bottle_missing_threshold and not throwing_detected:
            throwing_detected = True
            flash_counter = 0
            sound_played = False
            first_sound_finished = False
            sound_start_time = current_time

    # Display alert if throwing detected
    if throwing_detected:
        flash_counter += 1
        if flash_counter <= flash_duration:
            if flash_counter % 2 == 0:
                frame_resized = cv2.addWeighted(frame_resized, 0.7, alert_img, 0.3, 0)
                cv2.putText(frame_resized, "Throwing Detected!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                
                if not sound_played:
                    sound1.play()
                    sound_played = True
                elif not first_sound_finished and (current_time - sound_start_time) >= sound1_duration:
                    sound2.play()
                    first_sound_finished = True
        else:
            throwing_detected = False

    # Display bottle missing timer (for debugging)
    cv2.putText(frame_resized, f"Missing: {bottle_missing_duration:.1f}s", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("YOLO Pose Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
