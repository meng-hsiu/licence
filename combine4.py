import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image
import numpy as np
import time  # 用於延遲
import re  # 用於正則表達式

# 加載車輛偵測YOLO模型和車牌偵測YOLO模型
vehicle_model = YOLO("yolov10n.pt")
license_plate_model = YOLO("license_plate_detector.pt")

# 開啟攝像頭
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO模型偵測車輛
    results = vehicle_model(frame)

    detected_car_or_motorcycle = False

    # 處理每一個偵測結果
    for result in results:
        annotated_frame = result.plot()

        for bbox in result.boxes:
            confidence = bbox.conf[0]
            label = vehicle_model.names[int(bbox.cls)]

            # 偵測到車或摩托車且準確率為0.9以上
            if label.lower() in ['car', 'motorcycle'] and confidence >= 0.85:
                if label.lower() in ['car']:
                    print('car')
                else:
                    print('motorcycle')
                detected_car_or_motorcycle = True
                print(f"Detected {label} with confidence {confidence:.2f}")

                # 延遲2秒，讓鏡頭停留在當前畫面
                time.sleep(2)

                # 停留後擷取當前畫面
                ret, frame = cap.read()
                if not ret:
                    break

                # 使用車牌偵測模型進行偵測
                license_plate_results = license_plate_model(frame)
                detections = license_plate_results[0].boxes

                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # bounding box coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 設定框框顏色為綠色，粗細為2像素

                    # 裁剪偵測到的車牌區域
                    license_plate_img = frame[y1:y2, x1:x2]

                    # 將車牌圖像轉為灰階
                    gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

                    # 使用Canny邊緣檢測
                    edges = cv2.Canny(gray_plate, 50, 150, apertureSize=3)

                    # 使用霍夫直線變換來找出車牌的邊緣
                    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
                    angle = 0

                    if lines is not None:
                        for rho, theta in lines[0]:
                            angle = np.degrees(theta) - 90
                            break

                    # 旋轉車牌影像，使其變正
                    center = (license_plate_img.shape[1] // 2, license_plate_img.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_plate = cv2.warpAffine(license_plate_img, M, (license_plate_img.shape[1], license_plate_img.shape[0]))

                    # 調整對比度
                    alpha = 1.5
                    contrast_adjusted = cv2.convertScaleAbs(rotated_plate, alpha=alpha)

                    # OCR識別前的圖像處理
                    # sharp_img = cv2.Laplacian(gray_plate, cv2.CV_64F)
                    # sharp_img2 = cv2.convertScaleAbs(sharp_img)

                    _, binary_image = cv2.threshold(contrast_adjusted, 180, 255, cv2.THRESH_BINARY)

                    # 使用pytesseract進行OCR辨識
                    text = pytesseract.image_to_string(rotated_plate, lang='eng', config='--psm 11')
                    text_con = pytesseract.image_to_string(contrast_adjusted, lang='eng', config='--psm 11')
                    text_binary = pytesseract.image_to_string(binary_image, lang='eng', config='--psm 11')

                    # 正規化 去除不是A~Z或0~9的字
                    filtered_text = re.sub(r'[^A-Z0-9]', '', text.strip())
                    filtered_text_con = re.sub(r'[^A-Z0-9]', '', text_con.strip())
                    filtered_text_binary = re.sub(r'[^A-Z0-9]', '', text_binary.strip())

                    # 輸出偵測到的車牌號碼
                    print(f"Detected license plate number gray: {filtered_text.strip()}")
                    print(f"Detected license plate number con: {filtered_text_con.strip()}")
                    print(f"Detected license plate number binary: {filtered_text_binary.strip()}")

                    # 在車牌位置上顯示車牌號碼
                    cv2.putText(frame, text.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, text_con.strip(), (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 顯示影像
                    cv2.imshow("binary_image", binary_image)
                    cv2.imshow("contrast_adjusted", contrast_adjusted)
                    cv2.imshow("Rotated License Plate", rotated_plate)
                    cv2.imshow("Original Frame", frame)

                    # 等待鍵盤輸入並關閉視窗
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                # 重新開啟攝影機
                cap = cv2.VideoCapture(0)
                break

    # 顯示結果
        cv2.imshow('YOLO Webcam Detection with License Plate Recognition', annotated_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
