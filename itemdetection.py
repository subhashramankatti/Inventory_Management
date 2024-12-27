import torch
import cv2

# Load YOLOv5 model (small version for faster inference)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can use 'yolov5m', 'yolov5l', 'yolov5x' for larger models

# Function to perform object detection on video stream
def detect_objects_from_video():
    cap = cv2.VideoCapture(0)  # Open webcam (use video file path for video file input)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform inference with YOLOv5 model
        results = model(frame)

        # Render results on the image
        results.render()  # This adds bounding boxes to the image

        # Display the image with detections
        cv2.imshow("Detected Objects", results.ims[0])  # Display the first image in the batch

        # Press 'q' to exit the video loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
detect_objects_from_video()
