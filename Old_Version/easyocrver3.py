import cv2
from ultralytics import YOLO
import easyocr  # 引入 easyocr
import numpy as np
import time  # 用於延遲
import re  # 用於正則表達式
import os  # 用於儲存影像

# 加載車輛偵測YOLO模型和車牌偵測YOLO模型
vehicle_model = YOLO("../yolov10n.pt")
license_plate_model = YOLO("../license_plate_detector.pt")

# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 初始化 easyocr Reader
reader = easyocr.Reader(['en'])

# 設定車牌辨識功能狀態
recognition_enabled = False

def save_image(image, filename):
    path = os.path.join("../detected_plates", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
    print(f"Image saved as {filename}")

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

            # 偵測到車或摩托車且準確率為0.85以上
            if label.lower() in ['car', 'motorcycle'] and confidence >= 0.85:
                detected_car_or_motorcycle = True
                print(f"Detected {label} with confidence {confidence:.2f}")

                # 停留後擷取當前畫面
                ret, frame = cap.read()
                if not ret:
                    break

                if recognition_enabled:
                    # 使用車牌偵測模型進行偵測
                    license_plate_results = license_plate_model(frame)
                    detections = license_plate_results[0].boxes

                    for box in detections:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # bounding box coordinates
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        # 裁剪偵測到的車牌區域
                        license_plate_img = frame[y1:y2, x1:x2]

                        # 使用 easyocr 進行OCR辨識
                        ocr_results = reader.readtext(license_plate_img, detail=1)

                        for (bbox, text, ocr_confidence) in ocr_results:
                            filtered_text = re.sub(r'[^A-Z0-9]', '', text.strip())
                            print(f"Detected license plate: {filtered_text} with confidence {ocr_confidence:.2f}")

                            # 若OCR正確率超過0.8，儲存影像
                            if ocr_confidence > 0.8:
                                save_image(frame, f"{filtered_text}.jpg")

                        # 在車牌位置上顯示車牌號碼
                        cv2.putText(frame, filtered_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示結果
    cv2.imshow('YOLO Webcam Detection with License Plate Recognition', frame)

    # 按 'q' 鍵退出
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    # 按 'r' 鍵啟用或停用車牌辨識
    elif key & 0xFF == ord('r'):
        recognition_enabled = not recognition_enabled
        print(f"License plate recognition {'enabled' if recognition_enabled else 'disabled'}")

cap.release()
cv2.destroyAllWindows()
