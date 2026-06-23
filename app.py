import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # Object Detection + Tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    # Draw boxes and tracking IDs
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow(
        "Object Detection and Tracking",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()