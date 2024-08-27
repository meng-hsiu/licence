from ultralytics import YOLO
import pytesseract
import cv2

# Load pre-trained YOLO model
model = YOLO("license_plate_detector.pt")

# Perform object detection on an image
results = model("motor.jpg")
