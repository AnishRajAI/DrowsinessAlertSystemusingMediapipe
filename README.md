# Status: Local Deployment

## Note:
This project is designed to run locally because it requires direct access to the webcam and real-time hardware-based computation. Web hosting platforms cannot access local hardware directly, which makes full live deployment on the cloud infeasible.

Advantages of Local Deployment

Real-time detection: Direct hardware access ensures minimal latency for drowsiness detection.

Hardware compatibility: Can use device-specific functions like webcam and audio alerts.

High accuracy: Continuous frame analysis with MediaPipe and EAR calculations without network lag.

Safe for demos: Works reliably on your machine without depending on internet speed.

Future Deployment Options with React

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
To make this project globally accessible while supporting live webcam input, the project can be rewritten using React and MediaPipe JS:

Core Libraries / Tools Required

React – for frontend UI:

```npx create-react-app drowsiness-app```

MediaPipe Face Mesh (JS version) – for real-time face landmark detection:

```npm install @mediapipe/face_mesh @mediapipe/camera_utils @mediapipe/drawing_utils```

TensorFlow.js (if you want ML-based detection)

React hooks / state management – to handle EAR calculations, alerts

WebRTC / getUserMedia – for webcam streaming in the browser

Chart.js / Recharts / D3.js – for plotting live EAR/fatigue graphs

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Deployment Options

Netlify / Vercel: Simple static hosting for React apps

Docker + Cloud VM: If integrating Python backend for heavy processing

GitHub: Static demo without backend
