from ultralytics import YOLO
import pytesseract
import cv2

# Load pre-trained YOLO model
model = YOLO("license_plate_detector.pt")

# Perform object detection on an image
results = model("image2.jpg")

# Load the image
img = cv2.imread("image2.jpg")

# Get detected bounding boxes
detections = results[0].boxes

for box in detections:
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # bounding box coordinates

    # Crop the detected license plate area from the image
    license_plate_img = img[y1:y2, x1:x2]

    #blur
    img_blur = cv2.GaussianBlur(license_plate_img, (3, 3), 50)
    img_med = cv2.medianBlur(license_plate_img, 5)
    # Convert the cropped image to grayscale for better OCR results
    gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

    # Apply global thresholding
    _, plate_img_binary = cv2.threshold(gray_plate, 128, 255, cv2.THRESH_BINARY)

    # Apply adaptive thresholding
    plate_inv = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Apply OCR to extract text
    text = pytesseract.image_to_string(gray_plate, lang='eng', config='--psm 11')  # psm 8 for single word or line
    text_binary = pytesseract.image_to_string(plate_img_binary, lang='eng', config='--psm 11')
    text_inv = pytesseract.image_to_string(plate_inv, lang='eng', config='--psm 11')

    # Print detected text (license plate number)
    print(f"Detected license plate number: {text.strip()}")
    print(f"Detected license plate number (binary): {text_binary.strip()}")
    print(f"Detected license plate number (adaptive): {text_inv.strip()}")

    # Display the grayscale image
    cv2.imshow("License Plate", gray_plate)  # Use a string as the window name
    cv2.imshow("threshold", plate_img_binary)  # Use a string as the window name
    cv2.imshow("adaptive", plate_inv)  # Use a string as the window name
    cv2.imshow("med", img_med)  # Use a string as the window name
    cv2.imshow("blur", img_blur)  # Use a string as the window name

    # Wait for a key press and close the window
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()  # Close the window
