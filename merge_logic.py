import cv2
import numpy as np
import os
import math
from datetime import datetime
from ultralytics import YOLO
import cvzone
from tracker import Tracker

# --- Load YOLO models ---
model = YOLO("yolov8n.pt")   # for person, vehicle, traffic light
helmet_model = YOLO(r"C:\Users\HP\Desktop\cv_project\best_helmet.pt")  # helmet detector
helmet_class_names = {0: "Helmet", 1: "No Helmet"}

vehicle_classes = ["car", "motorcycle", "bus", "truck"]

# --- Helper functions ---
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Green & Red ranges
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
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

# Output folders
today_date = datetime.now().strftime('%Y-%m-%d')
traffic_dir = os.path.join("saved_images", today_date, "traffic_violation")
helmet_dir = os.path.join("saved_images", today_date, "helmet_violation")
os.makedirs(traffic_dir, exist_ok=True)
os.makedirs(helmet_dir, exist_ok=True)

saved_ids = []
current_light_status = None
stop_line_area = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    frame_count += 1

    # --- Run both YOLO models ---
    results = model(frame, conf=0.25, verbose=False)[0]
    results_helmet = helmet_model(frame, conf=0.25, verbose=False)[0]

    detections = []
    traffic_light_boxes = []
    person_boxes = []
    no_helmet_boxes = []

    # --- Main YOLO detections (person, vehicle, traffic light) ---
    for box in results.boxes:
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

        elif class_name == "person":
            person_boxes.append((x1, y1, x2, y2))

    # --- Helmet model detections ---
    for box in results_helmet.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.25:
            continue

        if class_id == 1:  # No Helmet
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            no_helmet_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, "No Helmet", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Helmet Violation Check ---
    for px1, py1, px2, py2 in person_boxes:
        person_top_center = ((px1 + px2) // 2, py1)
        violation = False

        for nx1, ny1, nx2, ny2 in no_helmet_boxes:
            no_helmet_center = get_center((nx1, ny1, nx2, ny2))
            dist = euclidean_dist(person_top_center, no_helmet_center)
            if dist < 50:  # threshold
                violation = True
                break

        if violation:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0,0,255), 2)
            cv2.putText(frame, "Helmet Violation", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(helmet_dir, f"frame_{frame_count}.jpg"), frame)

    # --- Traffic Light Stop-Line ---
    if traffic_light_boxes:
        _, _, _, light_y2 = traffic_light_boxes[0]
        stop_line_y = min(light_y2 + 100, frame.shape[0])
        stop_line_area = [
            (0, stop_line_y),
            (frame.shape[1], stop_line_y),
            (frame.shape[1], stop_line_y + 40),
            (0, stop_line_y + 40)
        ]

    # --- Vehicle Tracking ---
    tracked_vehicles = tracker.update(detections)

    for x3, y3, x4, y4, vid in tracked_vehicles:
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        if stop_line_area:
            result = cv2.pointPolygonTest(np.array(stop_line_area, np.int32), (cx, cy), False)
            if result >= 0:
                if current_light_status == "RED":
                    cvzone.putTextRect(frame, f"Traffic Violation {vid}", (x3, y3), 1, 1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

                    if vid not in saved_ids:
                        saved_ids.append(vid)
                        timestamp = datetime.now().strftime('%H-%M-%S')
                        cv2.imwrite(os.path.join(traffic_dir, f"{timestamp}.jpg"), frame)
                else:
                    cvzone.putTextRect(frame, f"{vid}", (x3, y3), 1, 1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

    # --- Draw Stop Line ---
    if stop_line_area:
        cv2.polylines(frame, [np.array(stop_line_area, np.int32)], True, (255, 255, 0), 2)

    cv2.imshow("Helmet + Traffic Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()






