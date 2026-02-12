from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import copy
from hand_detector import HandDetector, calc_landmark_list, pre_process_landmark

app = Flask(__name__)

# Load model and labels
model_path = 'model/keypoint_classifier/keypoint_classifier.pkl'
label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'

classifier = None
labels = []

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

if os.path.exists(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]

detector = HandDetector(num_hands=2)
current_translations = []

def generate_frames():
    global current_translations
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Original frame for processing (not flipped)
        input_frame = copy.deepcopy(frame)
        # Flipped frame for display
        display_frame = cv2.flip(frame, 1)
        
        # Detection
        results = detector.find_hands(input_frame)
        
        translations = []
        
        if results.hand_landmarks:
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                # Landmarks for prediction
                landmark_list = calc_landmark_list(input_frame, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Prediction
                hand_text = "..."
                if classifier is not None:
                    try:
                        prediction = classifier.predict([pre_processed_landmark_list])[0]
                        hand_text = labels[int(prediction)]
                    except:
                        hand_text = "Error"
                
                # Handedness
                handedness = results.handedness[i][0].category_name
                translations.append(f"{handedness}: {hand_text}")
                
                # Draw on display frame (mirrored)
                detector.draw_landmarks(display_frame, hand_landmarks, mirrored=True)
                
                # Overlay text
                h, w, c = display_frame.shape
                # Mirror the x coordinate for display text
                cx = int((1.0 - hand_landmarks[0].x) * w)
                cy = int(hand_landmarks[0].y * h)
                cv2.putText(display_frame, f"{handedness}: {hand_text}", (cx, cy - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        
        current_translations = translations
        
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/translations')
def get_translations():
    return jsonify(current_translations)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
