import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import winsound

EAR_THRESHOLD = 0.25
DROWSY_TIME = 3.0        
FATIGUE_ALERT_SCORE = 50
CAMERA_INDEX = 0

st.set_page_config(page_title="Drowsiness Detection (MediaPipe)", layout="centered")
st.title("Drowsiness Detection System (MediaPipe)")

def play_sound():
    try:
        winsound.Beep(1200, 800)
    except:
        pass

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([int(lm.x * w), int(lm.y * h)]))

    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)

if "detected" not in st.session_state:
    st.session_state.detected = False

if "fatigue_score" not in st.session_state:
    st.session_state.fatigue_score = 0

if "ear_history" not in st.session_state:
    st.session_state.ear_history = []

start = st.button("Start Monitoring")
frame_placeholder = st.empty()
status = st.empty()

col1, col2 = st.columns(2)
fatigue_metric = col1.metric("Fatigue Score", 0)
blink_metric = col2.metric("EAR (Live)", "0.00")

ear_chart = st.line_chart([])

if start and not st.session_state.detected:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("Cannot access camera")
        st.stop()

    eye_closed_start = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                st.session_state.fatigue_score += 2
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start >= DROWSY_TIME:
                    st.session_state.detected = True
                    play_sound()
                    status.error("DROWSINESS DETECTED")
                    break
            else:
                eye_closed_start = None
                st.session_state.fatigue_score = max(0, st.session_state.fatigue_score - 1)

            fatigue_metric.metric("Fatigue Score", st.session_state.fatigue_score)
            blink_metric.metric("EAR (Live)", f"{ear:.2f}")

            st.session_state.ear_history.append(ear)
            st.session_state.ear_history = st.session_state.ear_history[-100:]
            ear_chart.line_chart(st.session_state.ear_history)

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Fatigue: {st.session_state.fatigue_score}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if st.session_state.fatigue_score >= FATIGUE_ALERT_SCORE:
                st.session_state.detected = True
                play_sound()
                status.error("CAMERA STOPPED")
                break

        frame_placeholder.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

elif st.session_state.detected:
    st.success("Session ended. Refresh page to start again.")