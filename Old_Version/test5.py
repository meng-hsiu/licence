from ultralytics import YOLO
import pytesseract
import cv2
import numpy as np


def order_points(pts):
    """Order points in the following order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Apply perspective transform to get a top-down view of the license plate."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


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
    return np.mean(angles)


# Load pre-trained YOLO model
# model = YOLO("license_plate_detector.pt")
model = YOLO("car.pt")

# Perform object detection on an image
results = model("motor.jpg")

# Load the image
img = cv2.imread("../motor.jpg")

# Get detected bounding boxes
detections = results[0].boxes

for box in detections:
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # bounding box coordinates

    # Approximate the four corners of the license plate based on the bounding box
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")

    # Apply perspective transform to straighten the plate
    warped_plate = four_point_transform(img, +pts)

    # Convert the cropped image to grayscale for better OCR results
    gray_plate = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)

    # Detect edges for further rotation correction
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
    print(model.names)
    # Display the images
    cv2.imshow("Original License Plate", gray_plate)  # Use a string as the window name
    cv2.imshow("Rotated License Plate", rotated_plate)  # Use a string as the window name

    # Wait for a key press and close the window
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()  # Close the window
