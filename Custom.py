from ultralytics import YOLO
import pytesseract
from PIL import Image
import numpy as np
import cv2

# Load the YOLO model
model = YOLO("license_plate_detector.pt")

# Load the image using OpenCV
img_read = cv2.imread("image.jpg")

# Convert the OpenCV image (NumPy array) to a PIL image
img_pil = Image.fromarray(cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB))

# Resize the image using PIL
img_pil = img_pil.resize((300, 225), Image.LANCZOS)

# Convert the resized PIL image back to a NumPy array for YOLO model processing
img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 複製一個原始img拿來影像辨識用
img2 = np.copy(img)

# 將圖像轉換為灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

results = model(img2)

# Get detected bounding boxes
detections = results[0].boxes

for box in detections:
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # bounding box coordinates
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 設定框框顏色為綠色，粗細為2像素

    # Crop the detected license plate area from the image
    license_plate_img = img[y1:y2, x1:x2]
    license_plate_img_copy = np.copy(license_plate_img)
    cv2.imshow("license_plate_img_copy",license_plate_img_copy)

    # Convert the cropped NumPy array to a PIL image
    img_pil2 = Image.fromarray(cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2RGB))

    # Resize the image to 140x40
    image2 = img_pil2.resize((140, 40), Image.LANCZOS)

    # Convert the resized PIL image back to a NumPy array
    image = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)


    #blur
    img_blur = cv2.GaussianBlur(license_plate_img, (3, 3), 3)
    img_med = cv2.medianBlur(license_plate_img, 5)
    # Convert the cropped image to grayscale for better OCR results
    gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

    # 定義對比度增強的係數
    alpha = 4  # 對比度增強係數，越大越顯著
    beta = 0  # 亮度增強偏移量，這裡設為0表示只調整對比度

    # 對比度調整
    contrast_adjusted = cv2.convertScaleAbs(gray_plate, alpha=alpha, beta=beta)
    cv2.imshow("cont", contrast_adjusted)
    # 使用閉運算來填補字母內部的空隙
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(contrast_adjusted, cv2.MORPH_CLOSE, kernel)

    # Apply global thresholding
    _, plate_img_binary = cv2.threshold(gray_plate, 128, 255, cv2.THRESH_BINARY)

    # 使用掩碼將高亮度像素轉換為白色
    license_plate_img[contrast_adjusted > 200] = [255, 255, 255]  # 將高於閾值的像素設置為白色

    # Apply adaptive thresholding
    plate_inv = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Apply OCR to extract text
    text = pytesseract.image_to_string(gray_plate, lang='eng', config='--psm 11')  # psm 8 for single word or line
    text_binary = pytesseract.image_to_string(plate_img_binary, lang='eng', config='--psm 11')
    text_inv = pytesseract.image_to_string(plate_inv, lang='eng', config='--psm 11')
    text_cont = pytesseract.image_to_string(contrast_adjusted, lang='eng', config='--psm 11')


    # Print detected text (license plate number)
    print(f"Detected license plate number: {text.strip()}")
    print(f"Detected license plate number (binary): {text_binary.strip()}")
    print(f"Detected license plate number (adaptive): {text_inv.strip()}")
    print(f"Detected license plate number (contrast): {text_cont.strip()}")

    # Display the grayscale image
    cv2.imshow("license_plate_img", license_plate_img)
    cv2.imshow("closing", closing)
    cv2.imshow("Bigimg", image)
    cv2.imshow("Detected License Plates", img)
    cv2.imshow("org", img2)
    cv2.imshow("License Plate", gray_plate)  # Use a string as the window name
    cv2.imshow("threshold", plate_img_binary)  # Use a string as the window name
    cv2.imshow("adaptive", plate_inv)  # Use a string as the window name
    cv2.imshow("med", img_med)  # Use a string as the window name
    cv2.imshow("blur", img_blur)  # Use a string as the window name

    # Wait for a key press and close the window
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()  # Close the window


