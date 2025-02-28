import cv2
import pickle
import numpy as np
from flask import Flask, render_template, Response
import threading
import time

# Load parking positions from file
try:
    with open('park_positions', 'rb') as f:
        park_positions = pickle.load(f)
except FileNotFoundError:
    print("Error: Parking positions file not found.")
    park_positions = []

# Font for displaying text
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Threshold for determining empty spaces
empty_ratio = 0.05

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize status array with length equal to the number of parking slots
status_array = [0] * len(park_positions)  # 0 = empty, 1 = occupied

# Function to check if a car is in a parking zone
def is_car_in_zone(mask, zone):
    x, y, width, height = zone
    roi = mask[y:y + height, x:x + width]
    non_zero = cv2.countNonZero(roi)
    total_pixels = roi.size
    ratio = non_zero / total_pixels
    return ratio > empty_ratio

# Function to process parking spaces and update status array
def process_parking_spaces(frame, mask, overlay):
    for idx, position in enumerate(park_positions):
        if len(position) == 5: 
            spot_id, x, y, width, height = position
            zone = (x, y, width, height)

            # Check if any detection overlaps the parking zone
            car_in_zone = is_car_in_zone(mask, zone)

            # Update status based on car presence
            if car_in_zone:
                color = (0, 0, 255)  # Red for occupied
                status = 'occupied'
                status_array[idx] = 1  # Update status array
            else:
                color = (0, 255, 0)  # Green for empty
                status = 'empty'
                status_array[idx] = 0  # Update status array

            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
            cv2.putText(overlay, status, (x + 4, y + 20), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

# Function to update and print the status array every 2 seconds
def update_status_array():
    while True:
        print("Updated status array:", status_array)
        time.sleep(2)  # Wait for 2 seconds

# Main loop for video processing
cap = cv2.VideoCapture(0)  # cam 0
app = Flask('__name__')

def video_stream():
    while True:
        ret, frame = cap.read()
        if not ret:
            # Jika video selesai, reset ke frame awal
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue  # Lanjut ke iterasi berikutnya

        overlay = frame.copy()

        # Apply background subtraction
        fgmask = fgbg.apply(frame)

        # Process parking spaces and update status array
        process_parking_spaces(frame, fgmask, overlay)

        # Display the frame
        alpha = 0.7
        frame_new = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        ret, buffer = cv2.imencode('.jpeg', frame_new)
        frame = buffer.tobytes()
        yield (b' --frame\r\n' b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start a thread to update and print the status array every 2 seconds
    update_thread = threading.Thread(target=update_status_array)
    update_thread.daemon = True  # Thread akan berhenti saat program utama berhenti
    update_thread.start()

    # Run the Flask app
    app.run(host='0.0.0.0', port='5000', debug=False)