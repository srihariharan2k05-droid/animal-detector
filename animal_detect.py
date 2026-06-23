import streamlit as st
import cv2
from ultralytics import YOLO
import pygame
import time
import numpy as np

# Initialize Streamlit app
st.title("Animal & Human Detector - Streamlit Frontend")

# Sidebar configuration
st.sidebar.header("Settings")
run_detection = st.sidebar.checkbox("Start Detection")

# Load YOLO model
model = YOLO("yolo11n.pt")

# Initialize pygame
pygame.mixer.init()
alert_sound = "alert.mp3"

animal_classes = ["cat", "dog", "cow", "horse", "sheep", "elephant", "bear", "zebra", "giraffe", "bird"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

alarm_playing = False
alarm_start_time = 0

# Start camera
cap = cv2.VideoCapture(0)
stframe = st.empty()

while run_detection:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_regions = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Human", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        face_regions.append((x, y, x + w, y + h))

    results = model(frame)
    animal_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            skip = False
            for (fx1, fy1, fx2, fy2) in face_regions:
                if x1 > fx1 and y1 > fy1 and x2 < fx2 and y2 < fy2:
                    skip = True
                    break
            if skip:
                continue

            if class_name in animal_classes:
                animal_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{class_name} {conf * 100:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if animal_detected and not alarm_playing:
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play()
        alarm_start_time = time.time()
        alarm_playing = True

    if alarm_playing:
        elapsed = time.time() - alarm_start_time
        if elapsed >= 10:
            pygame.mixer.music.stop()
            alarm_playing = False

    stframe.image(frame, channels="BGR")

cap.release()
