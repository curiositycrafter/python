import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import math
from collections import deque
import random

# Verify model file
MODEL_PATH = 'emotion_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")

# Load model with error handling
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("🧠 Neural network loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

emotion_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Advanced emotion profiles with realistic colors and behaviors
emotion_profiles = {
    'angry': {
        'core': (0, 50, 255), 'mid': (30, 80, 255), 'outer': (60, 120, 255),
        'pulse_speed': 2.8, 'wave_chaos': 1.2, 'glow_intensity': 2.2,
        'particle_color': (100, 150, 255), 'energy': 0.9
    },
    'disgust': {
        'core': (40, 255, 100), 'mid': (80, 255, 140), 'outer': (120, 255, 180),
        'pulse_speed': 1.1, 'wave_chaos': 0.4, 'glow_intensity': 1.3,
        'particle_color': (150, 255, 200), 'energy': 0.4
    },
    'fear': {
        'core': (255, 0, 200), 'mid': (255, 60, 220), 'outer': (255, 120, 240),
        'pulse_speed': 4.2, 'wave_chaos': 1.8, 'glow_intensity': 2.8,
        'particle_color': (255, 150, 250), 'energy': 0.85
    },
    'happy': {
        'core': (0, 255, 255), 'mid': (80, 255, 255), 'outer': (160, 255, 255),
        'pulse_speed': 2.1, 'wave_chaos': 0.6, 'glow_intensity': 2.5,
        'particle_color': (200, 255, 255), 'energy': 0.8
    },
    'sad': {
        'core': (255, 150, 0), 'mid': (255, 180, 60), 'outer': (255, 210, 120),
        'pulse_speed': 0.7, 'wave_chaos': 0.2, 'glow_intensity': 1.1,
        'particle_color': (255, 200, 100), 'energy': 0.3
    },
    'surprise': {
        'core': (255, 255, 0), 'mid': (255, 255, 80), 'outer': (255, 255, 160),
        'pulse_speed': 5.5, 'wave_chaos': 2.2, 'glow_intensity': 3.2,
        'particle_color': (255, 255, 200), 'energy': 0.95
    },
    'neutral': {
        'core': (120, 180, 255), 'mid': (150, 200, 255), 'outer': (180, 220, 255),
        'pulse_speed': 1.2, 'wave_chaos': 0.3, 'glow_intensity': 1.4,
        'particle_color': (200, 230, 255), 'energy': 0.5
    }
}

# Initialize MediaPipe with optimized settings
mp_face = mp.solutions.face_detection
try:
    face_detection = mp_face.FaceDetection(
        model_selection=1,  # Full range model for better detection
        min_detection_confidence=0.7  # Higher confidence for stability
    )
    print("👁️  Advanced face detection initialized")
except Exception as e:
    raise RuntimeError(f"Failed to initialize MediaPipe: {e}")

class Particle:
    def __init__(self, x, y, color, energy=1.0):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2) * energy
        self.vy = random.uniform(-2, 2) * energy
        self.life = 1.0
        self.decay = random.uniform(0.005, 0.02)
        self.color = color
        self.size = random.uniform(1, 4)
        self.energy = energy
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        self.vx *= 0.98  # Air resistance
        self.vy *= 0.98
        return self.life > 0
    
    def draw(self, canvas):
        if self.life > 0:
            alpha = max(0, self.life)
            color = tuple(int(c * alpha) for c in self.color)
            size = max(1, int(self.size * alpha))
            cv2.circle(canvas, (int(self.x), int(self.y)), size, color, -1)

class JarvisStyleOrb:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.time_start = time.time()
        self.base_radius = 100
        self.particles = []
        self.energy_rings = []
        self.pulse_phase = 0
        self.wave_points = []
        
        # Initialize wave points around the orb
        for i in range(36):  # 36 points for smooth circle
            angle = (i / 36) * 2 * math.pi
            self.wave_points.append({
                'angle': angle,
                'radius_offset': 0,
                'phase': random.uniform(0, 2 * math.pi)
            })
    
    def create_deep_space_background(self, canvas):
        """Create a deep space-like background with gradients"""
        # Create radial gradient from center
        center = (self.center_x, self.center_y)
        max_radius = int(math.sqrt(self.width**2 + self.height**2) / 2)
        
        # Deep space gradient
        for radius in range(max_radius, 0, -8):
            progress = 1.0 - (radius / max_radius)
            # Deep blue to black gradient
            blue_intensity = int(25 * (1 - progress))
            green_intensity = int(15 * (1 - progress))
            red_intensity = int(10 * (1 - progress))
            
            color = (red_intensity, green_intensity, blue_intensity)
            cv2.circle(canvas, center, radius, color, -1)
        
        # Add some subtle stars
        current_time = time.time() - self.time_start
        for i in range(50):
            x = int((hash(f"star_x_{i}") % 10000) / 10000 * self.width)
            y = int((hash(f"star_y_{i}") % 10000) / 10000 * self.height)
            # Make stars twinkle
            brightness = abs(math.sin(current_time * 2 + i * 0.5)) * 100 + 50
            if brightness > 80:  # Only show bright stars
                cv2.circle(canvas, (x, y), 1, (int(brightness), int(brightness), int(brightness)), -1)
        
        return canvas
    
    def update_particles(self, canvas, emotion_profile, confidence):
        """Update and manage particle system"""
        # Remove dead particles
        self.particles = [p for p in self.particles if p.update()]
        
        # Add new particles based on emotion energy
        energy = emotion_profile['energy'] * confidence
        particle_count = int(energy * 8)
        
        for _ in range(particle_count):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(self.base_radius * 0.8, self.base_radius * 1.2)
            x = self.center_x + math.cos(angle) * distance
            y = self.center_y + math.sin(angle) * distance
            
            self.particles.append(Particle(
                x, y, 
                emotion_profile['particle_color'],
                energy
            ))
        
        # Draw particles
        for particle in self.particles:
            particle.draw(canvas)
    
    def create_energy_waves(self, canvas, emotion_profile, confidence):
        """Create dynamic energy wave patterns"""
        current_time = time.time() - self.time_start
        pulse_speed = emotion_profile['pulse_speed']
        wave_chaos = emotion_profile['wave_chaos']
        
        # Update wave points
        for point in self.wave_points:
            point['phase'] += pulse_speed * 0.02
            point['radius_offset'] = math.sin(point['phase']) * wave_chaos * 15 * confidence
        
        # Draw multiple energy rings
        for ring in range(3):
            ring_offset = ring * 25
            ring_alpha = 1.0 - (ring * 0.3)
            
            points = []
            for point in self.wave_points:
                radius = self.base_radius + ring_offset + point['radius_offset']
                x = self.center_x + math.cos(point['angle']) * radius
                y = self.center_y + math.sin(point['angle']) * radius
                points.append([int(x), int(y)])
            
            if len(points) > 2:
                points = np.array(points, np.int32)
                
                # Create color with transparency effect
                if ring == 0:  # Inner ring - brightest
                    color = emotion_profile['core']
                elif ring == 1:  # Middle ring
                    color = emotion_profile['mid']
                else:  # Outer ring
                    color = emotion_profile['outer']
                
                # Apply alpha
                color = tuple(int(c * ring_alpha * confidence) for c in color)
                
                # Draw the energy ring
                cv2.polylines(canvas, [points], True, color, 2)
    
    def create_core_orb(self, canvas, emotion_profile, confidence):
        """Create the main orb with advanced lighting effects"""
        current_time = time.time() - self.time_start
        pulse_speed = emotion_profile['pulse_speed']
        glow_intensity = emotion_profile['glow_intensity']
        
        # Dynamic radius based on emotion and time
        pulse = math.sin(current_time * pulse_speed) * 0.15 + 1
        radius = int(self.base_radius * (0.6 + confidence * 0.4) * pulse)
        
        # Create multiple layers for depth
        layers = [
            {'radius_mult': 1.0, 'color': emotion_profile['core'], 'alpha': 0.9},
            {'radius_mult': 0.8, 'color': emotion_profile['mid'], 'alpha': 0.7},
            {'radius_mult': 0.6, 'color': emotion_profile['core'], 'alpha': 0.5},
            {'radius_mult': 0.4, 'color': emotion_profile['mid'], 'alpha': 0.8},
            {'radius_mult': 0.2, 'color': emotion_profile['core'], 'alpha': 1.0}
        ]
        
        # Draw outer glow first
        glow_radius = int(radius * 1.8 * glow_intensity)
        for i in range(glow_radius, radius, -3):
            glow_alpha = 1.0 - ((i - radius) / (glow_radius - radius))
            glow_color = tuple(int(c * glow_alpha * 0.3 * confidence) for c in emotion_profile['outer'])
            cv2.circle(canvas, (self.center_x, self.center_y), i, glow_color, -1)
        
        # Draw main orb layers
        for layer in layers:
            layer_radius = int(radius * layer['radius_mult'])
            layer_color = tuple(int(c * layer['alpha'] * confidence) for c in layer['color'])
            cv2.circle(canvas, (self.center_x, self.center_y), layer_radius, layer_color, -1)
        
        # Add highlight effect
        highlight_offset_x = int(radius * 0.3)
        highlight_offset_y = int(radius * -0.3)
        highlight_pos = (self.center_x + highlight_offset_x, self.center_y + highlight_offset_y)
        highlight_radius = int(radius * 0.4)
        highlight_color = (255, 255, 255)
        
        # Create gradient highlight
        for i in range(highlight_radius, 0, -2):
            alpha = 1.0 - (i / highlight_radius)
            alpha_color = tuple(int(255 * alpha * 0.4) for _ in range(3))
            cv2.circle(canvas, highlight_pos, i, alpha_color, -1)
    
    def add_holographic_ui(self, canvas, emotion, confidence):
        """Add advanced holographic-style UI elements"""
        current_time = time.time() - self.time_start
        
        # Main emotion label with glowing effect
        label_text = emotion.upper()
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.2
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_x = self.center_x - text_width // 2
        text_y = 80
        
        # Glowing text background
        glow_padding = 20
        glow_alpha = abs(math.sin(current_time * 2)) * 0.3 + 0.2
        
        # Create multiple glow layers
        for i in range(5, 0, -1):
            glow_color = (int(100 * glow_alpha), int(150 * glow_alpha), int(255 * glow_alpha))
            cv2.rectangle(canvas,
                         (text_x - glow_padding - i, text_y - text_height - glow_padding - i),
                         (text_x + text_width + glow_padding + i, text_y + baseline + glow_padding + i),
                         glow_color, -1)
        
        # Main background
        cv2.rectangle(canvas,
                     (text_x - glow_padding, text_y - text_height - glow_padding),
                     (text_x + text_width + glow_padding, text_y + baseline + glow_padding),
                     (0, 0, 0), -1)
        
        # Main text
        cv2.putText(canvas, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Confidence indicator
        conf_text = f"CONFIDENCE: {confidence:.1%}"
        conf_font_scale = 0.6
        (conf_width, conf_height), _ = cv2.getTextSize(conf_text, font, conf_font_scale, 1)
        conf_x = self.center_x - conf_width // 2
        conf_y = text_y + 40
        
        cv2.putText(canvas, conf_text, (conf_x, conf_y), font, conf_font_scale, (150, 200, 255), 1)
        
        # Neural activity indicator
        activity_bars = 20
        bar_width = 3
        bar_spacing = 5
        start_x = self.center_x - (activity_bars * (bar_width + bar_spacing)) // 2
        base_y = self.height - 60
        
        for i in range(activity_bars):
            # Create dynamic bar heights based on time and emotion
            height_factor = abs(math.sin(current_time * 3 + i * 0.5)) * confidence
            bar_height = int(20 * height_factor + 5)
            
            bar_x = start_x + i * (bar_width + bar_spacing)
            bar_color = emotion_profiles[emotion]['particle_color']
            
            cv2.rectangle(canvas,
                         (bar_x, base_y - bar_height),
                         (bar_x + bar_width, base_y),
                         bar_color, -1)
        
        # Status indicators
        status_y = self.height - 30
        cv2.putText(canvas, "NEURAL NETWORK ACTIVE", (20, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
        
        # FPS counter
        fps_text = f"REAL-TIME PROCESSING"
        cv2.putText(canvas, fps_text, (self.width - 200, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

class AdvancedEmotionTracker:
    def __init__(self, smoothing=0.75):
        self.current_emotion = 'neutral'
        self.emotion_confidence = 0.5
        self.smoothing = smoothing
        self.emotion_history = deque(maxlen=20)
        self.confidence_history = deque(maxlen=10)
        self.transition_speed = 0.1
    
    def update(self, new_emotion, new_confidence):
        """Advanced emotion tracking with smooth transitions"""
        self.emotion_history.append((new_emotion, new_confidence))
        self.confidence_history.append(new_confidence)
        
        if len(self.emotion_history) >= 5:
            # Weighted recent emotion analysis
            recent_emotions = list(self.emotion_history)[-10:]
            emotion_weights = {}
            
            for i, (emotion, conf) in enumerate(recent_emotions):
                weight = (i + 1) * conf  # More recent and confident emotions have higher weight
                emotion_weights[emotion] = emotion_weights.get(emotion, 0) + weight
            
            # Get dominant emotion
            if emotion_weights:
                dominant_emotion = max(emotion_weights.items(), key=lambda x: x[1])[0]
                
                # Smooth transition
                if dominant_emotion != self.current_emotion:
                    # Only change if the new emotion is significantly stronger
                    current_weight = emotion_weights.get(self.current_emotion, 0)
                    new_weight = emotion_weights[dominant_emotion]
                    
                    if new_weight > current_weight * 1.3:  # 30% threshold for change
                        self.current_emotion = dominant_emotion
                
                # Smooth confidence
                avg_confidence = np.mean(list(self.confidence_history))
                self.emotion_confidence = (self.emotion_confidence * self.smoothing + 
                                         avg_confidence * (1 - self.smoothing))
        
        return self.current_emotion, self.emotion_confidence

# Initialize advanced systems
print("🚀 Initializing Jarvis-style emotion orb...")
orb_renderer = JarvisStyleOrb(800, 600)  # Higher resolution
emotion_tracker = AdvancedEmotionTracker()

# Initialize webcam with higher quality
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open webcam")

# Set higher resolution and frame rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Get actual resolution
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📹 Camera initialized: {actual_width}x{actual_height}")

print("🎭 JARVIS-STYLE EMOTION ORB ONLINE")
print("✨ Advanced neural emotion recognition active")
print("🔮 Holographic interface enabled")
print("⚡ Real-time particle physics simulation")
print("🎯 Press 'q' to terminate")

frame_count = 0
fps_start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Warning: Failed to capture frame")
            continue
        
        # Create high-resolution canvas
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        canvas = orb_renderer.create_deep_space_background(canvas)
        
        # Process frame for face detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        detected_emotion = 'neutral'
        detected_confidence = 0.3
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box with safety checks
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(w - x, int(bbox.width * w))
                height = min(h - y, int(bbox.height * h))
                
                if width > 50 and height > 50:  # Minimum face size
                    try:
                        # Extract and preprocess face
                        face_img = frame[y:y+height, x:x+width]
                        face_img = cv2.resize(face_img, (48, 48), 
                                            interpolation=cv2.INTER_CUBIC)
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        
                        # Normalize with better preprocessing
                        face_img = cv2.equalizeHist(face_img)  # Improve contrast
                        face_img = face_img.astype(np.float32) / 255.0
                        face_img = face_img.reshape(1, 48, 48, 1)
                        
                        # Predict emotion
                        pred = model.predict(face_img, verbose=0)
                        emotion_idx = np.argmax(pred)
                        detected_emotion = emotion_map[emotion_idx]
                        detected_confidence = float(pred[0][emotion_idx])
                        
                        break  # Use first valid detection
                        
                    except Exception as e:
                        print(f"⚠️  Face processing error: {e}")
                        continue
        
        # Update emotion tracking
        smooth_emotion, smooth_confidence = emotion_tracker.update(
            detected_emotion, detected_confidence
        )
        
        # Get emotion profile
        emotion_profile = emotion_profiles[smooth_emotion]
        
        # Render advanced orb components
        orb_renderer.create_energy_waves(canvas, emotion_profile, smooth_confidence)
        orb_renderer.update_particles(canvas, emotion_profile, smooth_confidence)
        orb_renderer.create_core_orb(canvas, emotion_profile, smooth_confidence)
        orb_renderer.add_holographic_ui(canvas, smooth_emotion, smooth_confidence)
        
        # Display the result
        cv2.imshow('JARVIS Emotion Orb - Neural Interface', canvas)
        
        # Calculate and display FPS occasionally
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - fps_start_time)
            print(f"🔄 FPS: {fps:.1f} | Emotion: {smooth_emotion.upper()} ({smooth_confidence:.1%})")
            fps_start_time = current_time
        
        # Exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

except KeyboardInterrupt:
    print("\n🛑 User interrupted")
except Exception as e:
    print(f"❌ Critical error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✨ JARVIS emotion orb terminated")
    print("🔮 Neural interface offline")