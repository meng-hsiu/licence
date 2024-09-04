import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image
import numpy as np
import time  # 用於延遲

# 加載車輛偵測YOLO模型和車牌偵測YOLO模型
vehicle_model = YOLO("../yolov8n.pt")
license_plate_model = YOLO("../license_plate_detector.pt")

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
                detected_car_or_motorcycle = True
                print(f"Detected {label} with confidence {confidence:.2f}")

                # 延遲2秒，讓鏡頭停留在當前畫面
                time.sleep(2)

                # 延遲後使用之前擷取的影像進行車牌偵測
                license_plate_results = license_plate_model(frame)
                detections = license_plate_results[0].boxes

                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # bounding box coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 設定框框顏色為綠色，粗細為2像素

                    # 裁剪偵測到的車牌區域
                    license_plate_img = frame[y1:y2, x1:x2]

                    # 將裁剪的NumPy圖像轉換為PIL圖像
                    img_pil = Image.fromarray(cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2RGB))

                    # 調整圖像大小為140x40
                    resized_plate_img = img_pil.resize((140, 40), Image.LANCZOS)

                    # 轉回NumPy陣列
                    resized_plate_img_np = cv2.cvtColor(np.array(resized_plate_img), cv2.COLOR_RGB2BGR)

                    # 將圖像轉換為灰階進行OCR辨識
                    gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

                    # 對比度調整 去污漬之類的
                    alpha = 2
                    contrast_adjusted = cv2.convertScaleAbs(gray_plate, alpha=alpha)
                    cv2.imshow('contrast_adjusted', contrast_adjusted)

                    # 使用pytesseract進行OCR辨識
                    text = pytesseract.image_to_string(contrast_adjusted, lang='eng', config='--psm 11')

                    # 輸出偵測到的車牌號碼
                    print(f"Detected license plate number: {text.strip()}")

                    # 顯示影像
                    cv2.imshow("Detected License Plate", frame)

                    # 等待鍵盤輸入並關閉視窗
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                break

    # 顯示結果
    cv2.imshow('YOLO Webcam Detection with License Plate Recognition', annotated_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
