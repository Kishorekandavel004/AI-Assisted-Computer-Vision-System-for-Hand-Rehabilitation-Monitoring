# AI-Assisted Computer Vision System for Hand Rehabilitation Monitoring

## 📌 Overview

An AI-assisted computer vision system for monitoring index-finger rehabilitation exercises using a standard webcam.

The system uses MediaPipe and OpenCV to detect hand landmarks and estimate index-finger joint movement. It measures movement range, counts repetitions, evaluates whether the target extension was achieved, records rehabilitation sessions, and uses machine learning to estimate next-session movement range.

---

## 🎯 Problem Statement

Post-stroke hand impairment can reduce finger movement, extension, and motor control. Conventional rehabilitation monitoring often requires direct supervision and manual assessment.

This project explores a low-cost computer-vision-based approach for monitoring repetitive index-finger rehabilitation exercises using a standard webcam.

---

## 💡 Proposed Solution

The system provides:

- Real-time hand detection
- Index-finger joint angle estimation
- Movement-range estimation
- Repetition counting
- Correct/incorrect repetition detection
- Session recording
- Progress visualization
- Machine-learning-based next-session ROM prediction

---

## 🧠 System Architecture

```text
Webcam
   ↓
MediaPipe Hand Landmark Detection
   ↓
Index Finger Landmark Extraction
   ↓
PIP Joint Angle Estimation
   ↓
Movement / ROM Analysis
   ↓
Repetition Detection
   ↓
Correct / Incorrect Evaluation
   ↓
Session Database
   ↓
Machine Learning
   ↓
Next-Session ROM Prediction
   ↓
Patient Dashboard
