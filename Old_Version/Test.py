import cv2
from ultralytics import YOLO
import pytesseract
# 加載預訓練的YOLO模型
'''model = YOLO('yolov8n.pt')'''  # 可以換成專門訓練過的車牌偵測模型
model = YOLO("../yolov8n.pt")

# 開啟攝像頭
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO模型進行偵測
    results = model(frame)

    # 處理每一個偵測結果
    for result in results:
        annotated_frame = result.plot()

        for bbox in result.boxes:
            label = model.names[int(bbox.cls)]



            # 假設模型標籤為 'license plate'
            if label.lower() in ['license plate', 'car plate', 'number plate']:
                x, y, w, h = map(int, bbox.xywh[0])
                plate_img = frame[y:y+h, x:x+w]

                # 影像預處理
                plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                _, plate_img_binary = cv2.threshold(plate_img_gray, 128, 255, cv2.THRESH_BINARY)

                # 使用pytesseract進行OCR辨識
                plate_text = pytesseract.image_to_string(plate_img_binary, config='--psm 8')
                print(f"Detected License Plate: {plate_text.strip()}")

    # 顯示結果
    cv2.imshow('YOLO Webcam Detection with License Plate Recognition', annotated_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


