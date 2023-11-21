import cv2
import time
from yolov4 import Detector

# Load the YOLO model
model = Detector(weights="../Yolo-Weights/yolov4.weights", cfg="../Yolo-Weights/yolov4.cfg", names="../Yolo-Weights/coco.names")

# Define the capture source (webcam or video file)
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture("car.mp4")  # For Video

classNames = model.get_object_names()

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    # Detect objects using YOLO
    results = model.detect(img)

    for r in results:
        x, y, w, h, label, conf = r
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)

        # Draw confidence and class label
        label_text = f'{label} {conf:.2f}'
        cv2.putText(img, label_text, (max(0, x), max(35, y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
