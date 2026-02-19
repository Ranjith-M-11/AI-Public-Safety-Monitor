🚨 AI Public Safety Monitor

Real-Time Threat Detection System using YOLOv8, Computer Vision, and Machine Learning

📌 Overview

AI Public Safety Monitor is an intelligent real-time surveillance system developed using Machine Learning and Computer Vision. The system uses the YOLOv8 object detection model to identify dangerous situations such as weapons, fire, fights, unknown persons, and restricted zone intrusions.

It automatically generates alerts, captures evidence, records videos, and logs incidents with timestamps. This system enhances public safety by providing automated threat monitoring and rapid response capability.

This project demonstrates practical implementation of real-time AI-based surveillance and intelligent monitoring systems.

🎯 Key Features

🔍 Real-Time Object Detection

1.Detects persons, weapons, and suspicious objects

2.Uses YOLOv8 deep learning model

3.High-speed and accurate detection

⚠️ Weapon Detection
Detects dangerous objects such as:

1.Knife

2.Bottle

3.Scissors

4.Suspicious carried objects

Automatically:

5.Saves screenshot

6.Sends alert

7.Logs incident

🔥 Fire Detection

1.Detects fire-like regions using image analysis

2.Triggers alerts instantly

3.Records evidence

🥊 Fight Detection

1.Detects aggressive motion patterns

2.Identifies possible fight situations

3.Uses movement and proximity analysis

👤 Person Tracking System

1.Assigns unique ID to each person

2.Tracks person movement across frames

3.Useful for surveillance and monitoring

Example:

Person ID: 1

Person ID: 2

🧠 Face Recognition System

1.Recognizes known persons

2.Detects unknown persons

3.Generates alert for unknown individuals

🚫 Restricted Zone Monitoring

1.Detects unauthorized entry into restricted area

2.Triggers instant alerts

3.Logs intrusion event

🎥 Automatic Video Recording

1.Automatically records video when threat detected

2.Stores video evidence

3.Saves in incident_logs/videos/

📊 Heatmap Visualization

1.Shows movement density of persons

2.Useful for behavior analysis

3.Helps identify crowded areas

📩 Alert System

1.Provides alerts via:

2.Sound alarm

3.Desktop notification

4.Telegram notification

5.Screenshot capture

6.Incident logging

🧠 Machine Learning Model Used

1.YOLOv8 (You Only Look Once v8)

2.Real-time object detection model

3.Pre-trained on COCO dataset

4.High accuracy and speed

5.Lightweight and efficient

Model file used:

-yolov8n.pt

🛠️ Technologies Used

1.Programming Language:

2.Python 3.11

Libraries and Frameworks:

1.OpenCV

2.Ultralytics YOLOv8

3.NumPy

4.Telebot (Telegram Bot API)

5.Plyer (Notifications)

Concepts:

1.Machine Learning

2.Computer Vision

3.Object Detection

4.Face Recognition

5.Motion Tracking

6.Real-Time Monitoring Systems

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

-Webcam captures real-time video

-YOLOv8 detects objects in each frame

-System analyzes behavior and object type

-If threat detected:

-Screenshot saved

-Video recorded

-Alert sent

-Incident logged

-System continues monitoring continuously


📈 Applications

-Public safety monitoring

-Smart surveillance systems

-Crime detection systems

-Campus security systems

-Smart city monitoring

-Industrial safety monitoring


🔒 Advantages

-Real-time detection

-Automated monitoring

-Offline system

-High accuracy

-Automatic evidence capture

Fast and efficient


🚀 Future Improvements

-Integration with CCTV cameras

-Cloud-based monitoring

-Mobile application support

-Advanced behavior prediction

-Multi-camera support

-AI crime prediction system

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
git commit -m "Added professional README"
git push
