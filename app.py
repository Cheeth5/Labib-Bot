from flask import Flask, render_template, redirect, url_for, request, session, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import time
import threading
import os
from datetime import datetime

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'your-secret-key-123'
app.config['UPLOAD_FOLDER'] = 'static/detections'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Shared detection data
detection_data = {
    'detection_count': 0,
    'latest_image': None,
    'last_detection': None
}

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'jetson':
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html', 
                         detection_count=detection_data['detection_count'],
                         image_file=detection_data['latest_image'],
                         detection_time=detection_data['last_detection'])

@app.route('/detections/<filename>')
def serve_detection(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

def run_detection():
    # Initialize models
    object_model = YOLO("yolov8n.pt").to('cuda')
    pose_model = YOLO("yolov8n-pose.pt").to('cuda')
    pygame.init()

    # Load resources
    alert_img = cv2.imread("static/alert.jpg")
    if alert_img is not None:
        alert_img = cv2.resize(alert_img, (640, 480))
    sound1 = pygame.mixer.Sound('static/0.mp3')
    sound2 = pygame.mixer.Sound('static/voice.mp3')

    # Detection variables
    previous_bottle_center = None
    last_bottle_time = None
    bottle_missing_threshold = 2  # 1.5 seconds threshold
    throwing_detected = False
    flash_duration = 30
    flash_counter = 0
    sound_played = False
    first_sound_finished = False
    photo_captured = False  # New flag to prevent multiple captures

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Throwing Detection System", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Throwing Detection System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_resized = cv2.resize(frame, (640, 480))
        current_time = time.time()
        display_frame = frame_resized.copy()

        # Object Detection
        bottle_center = None
        object_results = object_model(frame_resized, half=True)
        for result in object_results:
            for box in result.boxes.data:
                x1, y1, x2, y2, confidence, class_id = box.tolist()
                if int(class_id) == 39:  # Bottle class
                    bottle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Bottle", (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Pose Detection
        pose_results = pose_model(frame_resized, half=True, device=0)
        for result in pose_results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                for kp in keypoints:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)
                for i, j in skeleton:
                    if i < len(keypoints) and j < len(keypoints):
                        start = (int(keypoints[i][0]), int(keypoints[i][1]))
                        end = (int(keypoints[j][0]), int(keypoints[j][1]))
                        cv2.line(display_frame, start, end, (255, 0, 0), 2)

        # Throwing Detection Logic
        if bottle_center is not None:
            previous_bottle_center = bottle_center
            last_bottle_time = current_time
            throwing_detected = False
            photo_captured = False  # Reset capture flag when bottle reappears
            
            # Show timer countdown
            cv2.putText(display_frame, f"Bottle visible", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif last_bottle_time is not None:
            bottle_missing_duration = current_time - last_bottle_time
            remaining_time = max(0, bottle_missing_threshold - bottle_missing_duration)
            
            # Show remaining time
            cv2.putText(display_frame, f"Time left: {remaining_time:.1f}s", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if bottle_missing_duration >= bottle_missing_threshold and not throwing_detected and not photo_captured:
                throwing_detected = True
                photo_captured = True  # Set flag to prevent multiple captures
                detection_data['detection_count'] += 1
                detection_data['last_detection'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Save the detection image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detection_{timestamp}.jpg")
                cv2.imwrite(img_path, frame_resized)
                detection_data['latest_image'] = f"detection_{timestamp}.jpg"
                
                flash_counter = 0
                sound_played = False
                first_sound_finished = False
                sound_start_time = current_time
                sound1.play()

        # Display alert if throwing detected
        if throwing_detected:
            flash_counter += 1
            if flash_counter <= flash_duration:
                if flash_counter % 2 == 0 and alert_img is not None:
                    display_frame = cv2.addWeighted(display_frame, 0.7, alert_img, 0.3, 0)
                    cv2.putText(display_frame, "Throwing Detected!", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            else:
                throwing_detected = False

        # Add FPS counter (top-left corner)
        fps = 1 / (time.time() - current_time)
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the clean frame
        cv2.imshow("Throwing Detection System", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start detection in a separate thread
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.daemon = True
    detection_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)			
