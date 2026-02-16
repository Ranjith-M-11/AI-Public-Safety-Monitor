from ultralytics import YOLO
import cv2

# Load the lightweight YOLO model for webcam
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    annotated = results[0].plot()

    # Show detection window
    cv2.imshow("YOLO Webcam Test", annotated)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
