import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import winsound

EAR_THRESHOLD = 0.25
CLOSED_FRAMES_LIMIT = 30
ALARM_COOLDOWN = 3.0
CAMERA_INDEX = 0

st.set_page_config(page_title="Drowsiness Detection", layout="centered")
st.title(" Real-Time Drowsiness Detection (Local)")


def play_sound():
    try:
        winsound.Beep(1200, 700)
    except:
        pass

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(lm, eye, w, h):
    pts = [np.array([lm[i].x * w, lm[i].y * h]) for i in eye]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)

if "running" not in st.session_state:
    st.session_state.running = False
if "closed_frames" not in st.session_state:
    st.session_state.closed_frames = 0
if "last_alarm" not in st.session_state:
    st.session_state.last_alarm = 0

col1, col2 = st.columns(2)
if col1.button("Start"):
    st.session_state.running = True
if col2.button("Stop"):
    st.session_state.running = False
    st.session_state.closed_frames = 0

frame_box = st.empty()
status = st.empty()

ear_metric = st.metric("EAR", "0.00")
frame_metric = st.metric("Closed Frames", 0)

if st.session_state.running:
    cap = cv2.VideoCapture(CAMERA_INDEX)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            ear = (
                eye_aspect_ratio(lm, LEFT_EYE, w, h) +
                eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ) / 2.0

            ear_metric.metric("EAR", f"{ear:.2f}")

            if ear < EAR_THRESHOLD:
                st.session_state.closed_frames += 1
            else:
                st.session_state.closed_frames = 0

            frame_metric.metric("Closed Frames", st.session_state.closed_frames)

            if st.session_state.closed_frames >= CLOSED_FRAMES_LIMIT:
                if time.time() - st.session_state.last_alarm > ALARM_COOLDOWN:
                    play_sound()
                    st.session_state.last_alarm = time.time()
                    status.error("WAKE UP!")

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        frame_box.image(frame, channels="BGR")

    cap.release()
