from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
import threading
from queue import Queue
import time
import base64
import traceback
from collections import deque, Counter

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Load ASL model
try:
    model_path = os.path.join('asl_saved_model_finetuned_v3', 'asl_saved_model_finetuned_v3')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at: {model_path}")
    if not os.path.exists(os.path.join(model_path, 'saved_model.pb')):
        raise FileNotFoundError(f"Model file 'saved_model.pb' not found in: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Define ASL classes
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# Global variables
frame_queue = Queue(maxsize=2)
camera = None
is_running = False
current_sign = ""
prediction_history = deque(maxlen=15)  # Buffer for smoothing

def get_stable_prediction(predicted_class):
    """Returns the most frequent prediction in buffer"""
    prediction_history.append(predicted_class)
    most_common = Counter(prediction_history).most_common(1)
    if most_common:
        return most_common[0][0]
    return predicted_class

def process_frame(frame):
    """Process a single frame for ASL recognition"""
    global current_sign
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x_min, x_max, y_min, y_max = w, 0, h, 0
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)

                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)

                hand_region = rgb_frame[y_min:y_max, x_min:x_max]

                if model is not None and hand_region.size > 0:
                    try:
                        hand_region = cv2.resize(hand_region, (224, 224))
                        hand_region = hand_region.astype(np.float32) / 255.0
                        input_data = np.expand_dims(hand_region, axis=0)

                        with tf.device('/CPU:0'):
                            prediction = model.predict(input_data, verbose=0)

                        confidence = float(np.max(prediction[0]))
                        predicted_class = CLASSES[np.argmax(prediction[0])]

                        if confidence > 0.6:
                            stable_prediction = get_stable_prediction(predicted_class)
                            if stable_prediction != current_sign:
                                current_sign = stable_prediction
                                socketio.emit('sign_update', {
                                    'sign': current_sign,
                                    'confidence': confidence
                                })

                            text = f"Sign: {stable_prediction} ({confidence:.1%})"
                            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            cv2.rectangle(frame, (10, 10), (text_width + 20, 50), (0, 0, 0), -1)
                            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    except Exception as e:
                        print(f"Error during prediction: {str(e)}")
                        traceback.print_exc()
        else:
            # Reset prediction if no hand detected
            if current_sign != "":
                current_sign = ""
                prediction_history.clear()
                socketio.emit('sign_update', {'sign': "", 'confidence': 0.0})

        return frame
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return frame

def capture_frames():
    global camera, is_running
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    while is_running:
        success, frame = camera.read()
        if not success:
            print("Failed to grab frame")
            break

        processed_frame = process_frame(frame)

        if not frame_queue.full():
            frame_queue.put(processed_frame)
        else:
            frame_queue.get()
            frame_queue.put(processed_frame)

        time.sleep(0.01)

    camera.release()

def generate_frames():
    global is_running
    is_running = True
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()

    try:
        while is_running:
            if not frame_queue.empty():
                frame = frame_queue.get()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)
    except Exception as e:
        print(f"Error in generate_frames: {str(e)}")
    finally:
        is_running = False
        if camera is not None:
            camera.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

@app.route('/about-asl')
def about_asl():
    return render_template('about_asl.html')

@app.route('/about-product')
def about_product():
    return render_template('about_product.html')

@app.route('/helpful-links')
def helpful_links():
    return render_template('helpful_links.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/terms-of-service')
def terms_of_service():
    return render_template('terms_of_service.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('sign_update', {'sign': current_sign})

if __name__ == '__main__':
    print("Starting ASL Recognition Server...")
    print("Model loaded successfully!")
    print("Server will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        print("Trying alternative startup...")
        app.run(host='0.0.0.0', port=5000, debug=False)
