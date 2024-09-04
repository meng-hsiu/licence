import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import re

# 加載車輛偵測YOLO模型和車牌偵測YOLO模型
vehicle_model = YOLO("../yolov10n.pt")
license_plate_model = YOLO("../license_plate_detector.pt")

# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 初始化 easyocr Reader
reader = easyocr.Reader(['en'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO模型偵測車輛
    results = vehicle_model(frame)

    # 處理每一個偵測結果
    for result in results:
        annotated_frame = result.plot()

        for bbox in result.boxes:
            confidence = bbox.conf[0]
            label = vehicle_model.names[int(bbox.cls)]

            # 偵測到車或摩托車且準確率為0.85以上
            if label.lower() in ['car', 'motorcycle'] and confidence >= 0.85:
                # 使用車牌偵測模型進行偵測
                license_plate_results = license_plate_model(frame)
                detections = license_plate_results[0].boxes

                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 顯示車牌框框

                    # 裁剪偵測到的車牌區域
                    license_plate_img = frame[y1:y2, x1:x2]

                    # 進行OCR辨識
                    results = reader.readtext(license_plate_img)
                    text = ''.join([res[1] for res in results])

                    # 正規化 去除不是A~Z或0~9的字
                    filtered_text = re.sub(r'[^A-Z0-9]', '', text.strip())

                    # 在車牌位置上顯示車牌號碼
                    cv2.putText(frame, filtered_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示錄影畫面與偵測結果
    cv2.imshow('YOLO Webcam Detection with License Plate Recognition', frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
