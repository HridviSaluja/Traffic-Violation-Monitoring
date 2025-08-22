import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import cvzone
from tracker import Tracker

# --- Load YOLO ---
model = YOLO("yolov8n.pt")
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

# --- Traffic light color detection ---
def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Green range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Red range
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    green_pixels = cv2.countNonZero(mask_green)
    red_pixels = cv2.countNonZero(mask_red)

    if green_pixels > red_pixels and green_pixels > 10:
        return "GREEN", (0, 255, 0)
    elif red_pixels > green_pixels and red_pixels > 10:
        return "RED", (0, 0, 255)
    else:
        return None, None

# --- Prepare video ---
cap = cv2.VideoCapture("tr.mp4")
tracker = Tracker()
saved_ids = []

# Output folder
today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join("saved_images", today_date)
os.makedirs(output_dir, exist_ok=True)

current_light_status = None
stop_line_area = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    results = model(frame, conf=0.3, verbose=False)

    detections = []
    traffic_light_boxes = []

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name == "traffic light":
                roi = frame[y1:y2, x1:x2]
                label, color = detect_color(roi)
                if label:
                    current_light_status = label
                    traffic_light_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} light", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif class_name in vehicle_classes:
                detections.append([x1, y1, x2, y2])

    # --- Create dynamic stop-line area ---
    if traffic_light_boxes:
        # Pick first traffic light detected
        _, _, _, light_y2 = traffic_light_boxes[0]
        stop_line_y = min(light_y2 + 100, frame.shape[0])  # avoid overflow
        stop_line_area = [
            (0, stop_line_y),
            (frame.shape[1], stop_line_y),
            (frame.shape[1], stop_line_y + 40),
            (0, stop_line_y + 40)
        ]

    # --- Vehicle tracking ---
    tracked_vehicles = tracker.update(detections)

    for x3, y3, x4, y4, vid in tracked_vehicles:
        cx = int((x3 + x4) / 2)
        cy = int((y3 + y4) / 2)

        if stop_line_area:
            result = cv2.pointPolygonTest(np.array(stop_line_area, np.int32), (cx, cy), False)

            if result >= 0:
                if current_light_status == "RED":
                    cvzone.putTextRect(frame, f"Violation {vid}", (x3, y3), 1, 1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

                    if vid not in saved_ids:
                        saved_ids.append(vid)
                        timestamp = datetime.now().strftime('%H-%M-%S')
                        output_path = os.path.join(output_dir, f"{timestamp}.jpg")
                        cv2.imwrite(output_path, frame)
                else:
                    cvzone.putTextRect(frame, f"{vid}", (x3, y3), 1, 1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

    # Draw stop line
    if stop_line_area:
        cv2.polylines(frame, [np.array(stop_line_area, np.int32)], True, (255, 255, 0), 2)

    cv2.imshow("Traffic Light + Vehicle + Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



