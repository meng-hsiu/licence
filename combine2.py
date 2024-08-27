import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image
import numpy as np
import time  # 用於延遲

# 加載車輛偵測YOLO模型和車牌偵測YOLO模型
vehicle_model = YOLO("yolov8n.pt")
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
            if label.lower() in ['car', 'motorcycle'] and confidence >= 0.9:
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

                    # 高斯模糊
                    img_blur = cv2.GaussianBlur(license_plate_img, (3, 3), 3)

                    # 將裁剪的NumPy圖像轉換為PIL圖像
                    img_pil = Image.fromarray(cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2RGB))

                    # 調整圖像大小為140x40
                    resized_plate_img = img_pil.resize((140, 40), Image.LANCZOS)

                    # 轉回NumPy陣列
                    resized_plate_img_np = cv2.cvtColor(np.array(resized_plate_img), cv2.COLOR_RGB2BGR)

                    # 將圖像轉換為灰階進行OCR辨識
                    gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

                    # 調整對比度
                    alpha = 2
                    contrast_adjusted = cv2.convertScaleAbs(gray_plate, alpha=alpha)

                    # 使用拉普拉斯算子進行銳化
                    sharp_img = cv2.Laplacian(gray_plate, cv2.CV_64F)
                    sharp_img2 = cv2.convertScaleAbs(sharp_img)

                    # 使用pytesseract進行OCR辨識
                    text = pytesseract.image_to_string(gray_plate, lang='eng', config='--psm 11')
                    text_binary = pytesseract.image_to_string(contrast_adjusted, lang='eng', config='--psm 11')
                    text_sharp_img2 = pytesseract.image_to_string(sharp_img2, lang='eng', config='--psm 11')

                    # 輸出偵測到的車牌號碼
                    print(f"Detected license plate number: {text.strip()}")
                    print(f"Detected license plate number (contrast): {text_binary.strip()}")
                    print(f"Detected license plate number (contrast): {text_sharp_img2.strip()}")

                    # 在車牌位置上顯示車牌號碼
                    cv2.putText(frame, text_sharp_img2.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0),
                                2)
                    # 顯示影像
                    cv2.imshow("Original Frame", frame)
                    cv2.imshow("License Plate Area", license_plate_img)
                    cv2.imshow("Grayscale License Plate", gray_plate)
                    cv2.imshow("Contrast Adjusted License Plate", contrast_adjusted)

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
