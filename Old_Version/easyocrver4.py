import cv2
from ultralytics import YOLO
import easyocr
import re
import os

# 加載車輛偵測YOLO模型和車牌偵測YOLO模型
vehicle_model = YOLO("yolov10n.pt")
license_plate_model = YOLO("license_plate_detector.pt")

# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 初始化 easyocr Reader
reader = easyocr.Reader(['en'])

# 定義暫停偵測的變數
pause_detection = False

def save_image(image, filename):
    path = os.path.join("detected_plates", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
    # print(f"Image saved as {filename}")

# 存檔的方法
# def save_image(image, filename):
    # cv2.imwrite(filename, image)
    # print(f"Saved image as {filename}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or pause_detection:
        # 如果偵測暫停，等待按鍵按下來恢復偵測
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # 按下 'r' 鍵恢復偵測
            pause_detection = False
        continue

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
            if label.lower() in ['car', 'motorcycle'] and confidence >= 0.75:
                detected_car_or_motorcycle = True
                # print(f"Detected {label} with confidence {confidence:.2f}")

                # 使用車牌偵測模型進行偵測
                license_plate_results = license_plate_model(frame)
                detections = license_plate_results[0].boxes

                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # bounding box coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    license_plate_img = frame[y1:y2, x1:x2]

                    gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

                    # 使用 easyocr 進行 OCR 辨識
                    results = reader.readtext(gray_plate, detail=1)
                    for res in results:
                        text = res[1]
                        ocr_confidence = res[2]
                        filtered_text = re.sub(r'[^A-Z0-9]', '', text.strip())
                        print(f"Detected {label} with confidence {confidence:.2f} AND OCR detected text: {filtered_text} with confidence {ocr_confidence:.2f}")

                        # 若OCR正確率超過0.8，且filtered_text的長度在6到7個字之間，儲存影像
                        if ocr_confidence > 0.95 and 6 <= len(filtered_text) <= 7:
                            save_image(frame, f"{filtered_text}.jpg")
                            pause_detection = True  # 儲存後暫停偵測
                        else:
                            continue

                    # 顯示車牌號碼
                        cv2.putText(frame, filtered_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示結果
    cv2.imshow('YOLO Webcam Detection with License Plate Recognition', frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
