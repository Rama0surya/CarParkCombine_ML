import cv2
import pickle
import requests
import numpy
from ultralytics import YOLO
from flask import Flask, render_template, Response, stream_with_context, request

# Initialize YOLO model
yolo_model = YOLO("best.pt")

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
empty_ratio = 0.20

# Function to check if a car is in a parking zone
def is_car_in_zone(detection, zone):

    x1, y1, x2, y2 = detection
    zx, zy, zw, zh = zone
    # Check for overlap
    return x1 < zx + zw and x2 > zx and y1 < zy + zh and y2 > zy

# Function to process parking spaces
def process_parking_spaces(frame, detections, overlay):
    for position in park_positions:
        if len(position) == 5: 
            spot_id, x, y, width, height = position
            zone = (x, y, width, height)

            # Check if any detection overlaps the parking zone
            car_in_zone = False
            for detection in detections:
                if detection['class'] == 2: 
                    bbox = detection['box']
                    if is_car_in_zone(bbox, zone):
                        car_in_zone = True
                        break

            # Update status based on car presence
            if car_in_zone:
                color = (0, 0, 255)  # Red for occupied
                status = 'occupied'
                # url = f"http://localhost:3000/updateStatusCam/mobil{spot_id}?status=true"
            else:
                color = (0, 255, 0)  # Green for empty
                status = 'empty'
                # url = f"http://localhost:3000/updateStatusCam/mobil{spot_id}?status=false"

            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
            cv2.putText(overlay, status, (x + 4, y + 20), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            # Send HTTP request
            # try:
            #     response = requests.get(url)
            #     print(f"Sent request to {url}, response: {response.status_code}")
            # except requests.RequestException as e:
            #     print(f"Failed to send request to {url}: {e}")

# Main loop for video processing
cap = cv2.VideoCapture('carPark.mp4')  # cam 0
app = Flask('__name__')

while True:
    ret, frame = cap.read()
    if not ret:
            break;
        else:
            ret, buffer = cv2.imencode('.jpeg',frame)
            frame = buffer.tobytes()
            yield (b' --frame\r\n' b'Content-type: imgae/jpeg\r\n\r\n' + frame +b'\r\n')

    overlay = frame.copy()

    # Perform YOLO detection
    results = yolo_model(frame)
    detections = [
        {
            "box": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            "class": int(cls),
            "confidence": conf
        }
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf)
    ]

    # Process parking spaces
    process_parking_spaces(frame, detections, overlay)

    # Display the frame
    alpha = 0.7
    frame_new = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.imshow('frame', frame_new)

    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(host='0.0.0.0', port='5000', debug=False)
