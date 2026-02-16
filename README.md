🚨 AI Public Safety Monitor
Real-Time Threat Detection System using YOLOv8, Computer Vision, and Machine Learning

📌 Overview

AI Public Safety Monitor is an intelligent real-time surveillance system developed using Machine Learning and Computer Vision. The system uses the YOLOv8 object detection model to identify dangerous situations such as weapons, fire, fights, unknown persons, and restricted zone intrusions.

It automatically generates alerts, captures evidence, records videos, and logs incidents with timestamps. This system enhances public safety by providing automated threat monitoring and rapid response capability.

This project demonstrates practical implementation of real-time AI-based surveillance and intelligent monitoring systems.

🎯 Key Features

🔍 Real-Time Object Detection

Detects persons, weapons, and suspicious objects

Uses YOLOv8 deep learning model

High-speed and accurate detection


⚠️ Weapon Detection

Detects dangerous objects such as:

Knife

Bottle

Scissors

Suspicious carried objects


Automatically:

Saves screenshot

Sends alert

Logs incident


🔥 Fire Detection

Detects fire-like regions using image analysis

Triggers alerts instantly

Records evidence


🥊 Fight Detection

Detects aggressive motion patterns

Identifies possible fight situations

Uses movement and proximity analysis


👤 Person Tracking System

Assigns unique ID to each person

Tracks person movement across frames

Useful for surveillance and monitoring


Example:

Person ID: 1
Person ID: 2

🧠 Face Recognition System

Recognizes known persons

Detects unknown persons

Generates alert for unknown individuals

🚫 Restricted Zone Monitoring

Detects unauthorized entry into restricted area

Triggers instant alerts

Logs intrusion event


🎥 Automatic Video Recording

Automatically records video when threat detected

Stores video evidence

Saves in incident_logs/videos/


📊 Heatmap Visualization

Shows movement density of persons

Useful for behavior analysis

Helps identify crowded areas


📩 Alert System

Provides alerts via:

Sound alarm

Desktop notification

Telegram notification

Screenshot capture

Incident logging


🧠 Machine Learning Model Used
YOLOv8 (You Only Look Once v8)

Real-time object detection model

Pre-trained on COCO dataset

High accuracy and speed

Lightweight and efficient

Model file used:

yolov8n.pt


🛠️ Technologies Used

Programming Language:

Python 3.11

Libraries and Frameworks:

OpenCV

Ultralytics YOLOv8

NumPy

Telebot (Telegram Bot API)

Plyer (Notifications)


Concepts:

Machine Learning

Computer Vision

Object Detection

Face Recognition

Motion Tracking

Real-Time Monitoring Systems


📂 Project Structure
AI-Public-Safety-Monitor/
│
├── src/
│   ├── main.py
│   ├── audio_monitor.py
│   ├── face_train.py
│
├── incident_logs/
│   ├── screenshots/
│   ├── videos/
│   ├── events_log.csv
│
├── telegram_config.json
├── requirements.txt
└── README.md

▶️ Installation and Setup
Step 1: Clone Repository
git clone https://github.com/Ranjith-M-11/AI-Public-Safety-Monitor.git

Step 2: Navigate to Project Folder
cd AI-Public-Safety-Monitor

Step 3: Create Virtual Environment (Recommended)
python -m venv venv


Activate:

venv\Scripts\activate

Step 4: Install Dependencies
pip install -r requirements.txt

Step 5: Run the Project
python src/main.py


⚙️ How It Works

Webcam captures real-time video

YOLOv8 detects objects in each frame

System analyzes behavior and object type

If threat detected:

Screenshot saved

Video recorded

Alert sent

Incident logged

System continues monitoring continuously


📈 Applications

Public safety monitoring

Smart surveillance systems

Crime detection systems

Campus security systems

Smart city monitoring

Industrial safety monitoring


🔒 Advantages

Real-time detection

Automated monitoring

Offline system

High accuracy

Automatic evidence capture

Fast and efficient


🚀 Future Improvements

Integration with CCTV cameras

Cloud-based monitoring

Mobile application support

Advanced behavior prediction

Multi-camera support

AI crime prediction system

👨‍💻 Author

Ranjith M
B.E Artificial Intelligence
Machine Learning Project

GitHub:
https://github.com/Ranjith-M-11

⭐ Project Status

✅ Completed
✅ Fully Functional
✅ Real-Time ML System
✅ Ready for Deployment

🧠 Keywords

Machine Learning, YOLOv8, Computer Vision, Surveillance System, Object Detection, AI Security System, Python, OpenCV

✅ After pasting this

Run:

git add README.md
git commit -m "Added professional README"
git push
