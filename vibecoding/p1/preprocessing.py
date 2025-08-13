import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_fer2013(data_dir='imgs/images'):
    images = []
    labels = []
    emotion_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    
    for emotion_idx, emotion in emotion_map.items():
        folder = os.path.join(data_dir, 'train', emotion)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(emotion_idx)
    
    return np.array(images), np.array(labels)

# Load and preprocess
X, y = load_fer2013()
X = X.reshape(X.shape[0], 48, 48, 1)  # Add channel dimension
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)