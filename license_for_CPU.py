import cv2
import easyocr
import numpy as np

# 初始化 EasyOCR 讀取器
reader = easyocr.Reader(['en'])

def detect_license_plate_from_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影像，請檢查攝影機。")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred_frame, 100, 200)

        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        license_plate_found = False

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:  # 假設找到的四邊形是車牌
                x, y, w, h = cv2.boundingRect(approx)
                license_plate_img = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 轉換為灰階並進行二值化
                gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)
                _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 進行 OCR 辨識
                results_text = reader.readtext(binary_plate, detail=1)
                if results_text:
                    for (bbox_text, text, prob) in results_text:
                        print(f"識別的文字: {text} (信心: {prob:.2f})")
                        if prob > 0.85 and 6 <= len(text) <= 7:
                            print(f"符合條件的車牌: {text}")
                            license_plate_found = True
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                            break

            if license_plate_found:
                break

        cv2.imshow('License Plate Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_license_plate_from_camera()
