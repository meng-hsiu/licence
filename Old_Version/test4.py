from ultralytics import YOLO
import pytesseract
import cv2
import numpy as np


def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image


def get_angle_from_lines(lines):
    """Calculate the angle of rotation based on detected lines."""
    if lines is None:
        return 0

    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)

    # Average angle of all detected lines
    return np.mean(angles)


# Load pre-trained YOLO model
model = YOLO("../license_plate_detector.pt")

# Perform object detection on an image
results = model("image2.jpg")

# Load the image
img = cv2.imread("../image2.jpg")

# Get detected bounding boxes
detections = results[0].boxes

for box in detections:
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # bounding box coordinates

    # Crop the detected license plate area from the image
    license_plate_img = img[y1:y2, x1:x2]

    # Convert the cropped image to grayscale for better OCR results
    gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray_plate, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # Get the angle from the detected lines
    angle = get_angle_from_lines(lines)

    # Rotate the image to correct the angle
    rotated_plate = rotate_image(gray_plate, +angle)

    # Apply OCR to extract text
    text = pytesseract.image_to_string(rotated_plate, config='--psm 6')  # psm 6 for a block of text

    # Print detected text (license plate number)
    print(f"Detected license plate number: {text.strip()}")

    # Display the images
    cv2.imshow("Original License Plate", gray_plate)  # Use a string as the window name
    cv2.imshow("Rotated License Plate", rotated_plate)  # Use a string as the window name

    # Wait for a key press and close the window
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()  # Close the window
