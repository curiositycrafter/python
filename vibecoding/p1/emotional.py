import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('emotion_model.h5')
emotion_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            
            # Extract face region
            face_img = frame[y:y+height, x:x+width]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Fixed: COLOR_BGR2GRAYSCALE → COLOR_BGR2GRAY
            face_img = face_img / 255.0
            face_img = face_img.reshape(1, 48, 48, 1)
            
            # Predict emotion
            pred = model.predict(face_img)
            emotion_idx = np.argmax(pred)
            emotion = emotion_map[emotion_idx]
            confidence = pred[0][emotion_idx]
            
            # Display emotion (temporary, will replace with avatar)
            cv2.putText(frame, f'{emotion} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()