import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('emotion_model.h5')

# Load preprocessed validation data
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

# Print the accuracy
print(f'Validation Accuracy: {accuracy * 100:.2f}%')