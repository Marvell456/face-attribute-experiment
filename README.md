# Experimental Real-Time Face Recognition & Attribute Analysis  
*A small experimental project for exploring the limits of open-source face analysis models*

## Overview

This project is a **real-time computer vision experiment** I made to explore on the capabilities and limitations of open-source face recognition and attribute models when applied to live webcam footage.

Using a live webcam feed, the system:
- detects and recognizes known faces
- estimates age, gender, and emotion
- approximates distance from the camera
- adapts to changing lighting conditions in real time

The goal of this project was **not accuracy or deployment**, but for me to explore on how these models behave under real-world conditions and how I can calibrate them to be more accurate.

## Project Context

This project was built **before** my *Face Recognition Attendance System* as a small experimental project meant to explore more on open sourced face ML libraries.

Before creating a serious full-stack, real-time system, I wanted to experiment first with open-source face recognition and attribute models (deepface, face_recognition, etc) to understand their strengths, and weaknesses when exposed to real time video.

The insights gained here directly influenced the design decisions in my later, system-based projects.

## What This Project Does

- **Face recognition** using `face_recognition` (dlib embeddings)
- **Attribute estimation** (age, gender, emotion) using DeepFace
- **Distance approximation** based on face bounding box size
- **Lighting robustness experiments** using:
  - adaptive gamma correction
  - CLAHE contrast enhancement
- **Real-time performance** achieved using multithreading to avoid blocking the video feed

All processing runs **locally on-device** in real time using a standard webcam.

## Key Experiments & Observations

Through testing and experimentation, I observed that:

- Attribute predictions (age, emotion) are extremely unstable especially with changes in lighting and head pose
- Recognition accuracy drops when face is not shown entirely, or is tilted
- Small preprocessing changes has a positive affect model stability
- Freezes in the system are more often caused by **I/O and inference (ML models)**, not rendering (cv2 drawing boxes)

This project helped me understand that **machine learning performance in practice depends more on system design than on model choice alone**.

## Important Limitations (Intentional)

This project is **not production-ready** and does not attempt to be.

Known limitations include:
- No dataset balancing or demographic calibration
- Emotion and age estimates are approximate and unstable
- Distance estimation is rough and camera-dependent
- No privacy controls or deployment safeguards
- Designed strictly for experimentation, not decision-making

These limitations are intentionally documented as part of the learning process.  

## Why This Project Matters to Me

This project taught me to:
- treat ML outputs as *signals*, not ground truth
- question model confidence instead of trusting predictions blindly
- evaluate systems under imperfect, real-world conditions
- build intuition before building deployable products

It directly shaped how I later approached system design with a stronger focus on reliability, privacy, and practical usability.

---

## Tech Stack

- Python
- OpenCV
- face_recognition (dlib)
- DeepFace
- NumPy
- Multithreading (Python)

---

## Disclaimer

This project is intended for **educational and experimental purposes only**.  
Face analysis outputs is not recommended to be implemented for serious real world applications.
