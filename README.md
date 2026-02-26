# Vision-Guided Robot Arm

Vision-guided pick-and-place robotic arm using YOLO detection, camera-to-world calibration, and XML-RPC-based robot control.

## Architecture

Camera → YOLO detection → Pixel-to-mm transform → Robot motion via XML-RPC

## Features

- Real-time object detection
- Perspective calibration
- Safe Z positioning
- Pick-and-place logic

## Tech Stack

- Python
- OpenCV
- YOLO
- Fairino SDK
- XML-RPC