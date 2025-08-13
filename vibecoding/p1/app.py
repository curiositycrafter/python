import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading

app = Flask(__name__, template_folder='.', static_folder='.')

# Load emotion model
try:
    model = tf.keras.models.load_model('emotion_model.h5')
    print("Emotion model loaded successfully")
except Exception as e:
    print(f"Error loading emotion_model.h5: {e}")
emotion_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
current_emotion_lock = threading.Lock()
current_emotion = {'emotion': 'neutral', 'confidence': 0.0}

# Initialize MediaPipe
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

def webcam_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam")
            break
        print(f"Frame captured: {frame.shape}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections:
            print(f"Detected {len(results.detections)} face(s)")
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                face_img = frame[y:y+height, x:x+width]
                if face_img.size == 0:
                    print("Warning: Empty face image detected")
                    continue
                face_img = cv2.resize(face_img, (48, 48))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_img = face_img / 255.0
                face_img = face_img.reshape(1, 48, 48, 1)
                pred = model.predict(face_img)
                emotion_idx = np.argmax(pred)
                with current_emotion_lock:
                    global current_emotion
                    current_emotion = {'emotion': emotion_map[emotion_idx], 'confidence': float(pred[0][emotion_idx])}
                    print(f"Updated emotion: {current_emotion}")
        else:
            print("No faces detected")
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_emotion')
def get_emotion():
    with current_emotion_lock:
        print(f"Serving emotion: {current_emotion}")
        return jsonify(current_emotion)

@app.route('/cyber_face.glb')
def serve_model():
    if not os.path.exists('cyber_face.glb'):
        print("Error: cyber_face.glb not found in project directory")
        return "Model file not found", 404
    return send_file('cyber_face.glb')

if __name__ == '__main__':
    print("Starting Flask server...")
    threading.Thread(target=webcam_thread, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5000)