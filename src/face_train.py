import cv2
import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FACES_DIR = os.path.join(ROOT_DIR, "faces")
MODEL_PATH = os.path.join(ROOT_DIR, "face_model.yml")
LABELS_PATH = os.path.join(ROOT_DIR, "face_labels.pkl")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
labels = {}
x_train = []
y_labels = []

print("👤 Scanning faces folder:", FACES_DIR)

for root, dirs, files in os.walk(FACES_DIR):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_")

            if label not in labels:
                labels[label] = current_id
                current_id += 1

            id_ = labels[label]

            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)

print(f"✅ Collected {len(x_train)} face samples.")

if not x_train:
    print("❌ No faces found. Check your faces/ folder images.")
    raise SystemExit

recognizer.train(x_train, np.array(y_labels))
recognizer.write(MODEL_PATH)

with open(LABELS_PATH, "wb") as f:
    pickle.dump(labels, f)

print("🎉 Training complete! Model saved:")
print("   ", MODEL_PATH)
print("   ", LABELS_PATH)
