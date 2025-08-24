import cv2
import numpy as np
import math
import tempfile
from ultralytics import YOLO
import cvzone
from tracker import Tracker
import streamlit as st
from datetime import datetime

# --- Streamlit UI ---
st.set_page_config(page_title="Helmet & Traffic Violation Detection", layout="wide")
st.title("🚦 Helmet & Traffic Violation Detection")
st.markdown("Upload a video to detect **helmet violations** and **traffic signal violations**.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.2, 1.0, 0.25, 0.05)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)

# Video upload
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# Load models
model = YOLO("yolov8n.pt")
helmet_model = YOLO(r"C:\Users\HP\Desktop\cv_project\best_helmet.pt")
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

# Helper functions
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    if cv2.countNonZero(mask_green) > 20:
        return "GREEN", (0, 255, 0)
    elif cv2.countNonZero(mask_red) > 20:
        return "RED", (0, 0, 255)
    return None, None

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    progress = st.progress(0)
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("📊 Detection Summary")
        helmet_violation_count = st.empty()
        traffic_violation_count = st.empty()
        frame_counter = st.empty()

    tracker = Tracker()
    stop_line_area = None
    current_light_status = None
    frame_count = 0
    helmet_violations = 0
    traffic_violations = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (1280, 720))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame_count += 1

        results = model(frame, conf=conf_threshold, verbose=False)[0]
        results_helmet = helmet_model(frame, conf=conf_threshold, verbose=False)[0]

        detections, person_boxes, traffic_light_boxes, no_helmet_boxes = [], [], [], []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name == "traffic light":
                roi = frame[y1:y2, x1:x2]
                label, color = detect_color(roi)
                if label:
                    current_light_status = label
                    traffic_light_boxes.append((x1, y1, x2, y2))
                    if show_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} light", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif name in vehicle_classes:
                detections.append([x1, y1, x2, y2])
            elif name == "person":
                person_boxes.append((x1, y1, x2, y2))

        for box in results_helmet.boxes:
            if int(box.cls[0]) == 1:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                no_helmet_boxes.append((x1, y1, x2, y2))

        for px1, py1, px2, py2 in person_boxes:
            person_top = ((px1 + px2) // 2, py1)
            for nx1, ny1, nx2, ny2 in no_helmet_boxes:
                if euclidean_dist(person_top, get_center((nx1, ny1, nx2, ny2))) < 50:
                    helmet_violations += 1
                    if show_boxes:
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        cv2.putText(frame, "Helmet Violation", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if traffic_light_boxes:
            _, _, _, light_y2 = traffic_light_boxes[0]
            stop_line_y = min(light_y2 + 100, frame.shape[0])
            stop_line_area = [(0, stop_line_y), (frame.shape[1], stop_line_y), (frame.shape[1], stop_line_y + 40), (0, stop_line_y + 40)]

        tracked_vehicles = tracker.update(detections)
        for x3, y3, x4, y4, vid in tracked_vehicles:
            cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
            if stop_line_area:
                result = cv2.pointPolygonTest(np.array(stop_line_area, np.int32), (cx, cy), False)
                if result >= 0 and current_light_status == "RED":
                    traffic_violations += 1
                    if show_boxes:
                        cvzone.putTextRect(frame, f"Traffic Violation {vid}", (x3, y3), 1, 1)

        if stop_line_area:
            cv2.polylines(frame, [np.array(stop_line_area, np.int32)], True, (255, 255, 0), 2)

        out.write(frame)
        stframe.image(frame, channels="BGR")  # ✅ FIXED (removed use_container_width)

        progress.progress(frame_count / total_frames)
        helmet_violation_count.metric("Helmet Violations", helmet_violations)
        traffic_violation_count.metric("Traffic Violations", traffic_violations)
        frame_counter.metric("Frames Processed", frame_count)

    cap.release()
    out.release()

    st.success("✅ Processing complete!")
    st.download_button("Download Processed Video", open(out_path, "rb"), "processed_video.mp4")

