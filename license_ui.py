# Form implementation generated from reading ui file 'license.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal
import cv2
from ultralytics import YOLO
import easyocr
import re
import os
import pyodbc

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1118, 435)
        self.label = QtWidgets.QLabel(parent=Dialog)
        self.label.setGeometry(QtCore.QRect(80, 60, 971, 271))
        font = QtGui.QFont()
        font.setPointSize(50)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "車牌辨識"))
        self.label.setText(_translate("Dialog", "正在等待偵測..."))

class VideoCaptureThread(QThread):
    text_detected = pyqtSignal(str)

    vehicle_model = YOLO("yolov10n.pt")
    license_plate_model = YOLO("license_plate_detector.pt")

    @staticmethod
    def save_image(image, filename):
        path = os.path.join("detected_plates", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        reader = easyocr.Reader(['en'])
        pause_detection = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            if pause_detection:
                key = cv2.waitKey(1)
                if key == ord('r'):
                    pause_detection = False
                continue

            results = self.vehicle_model(frame)
            detected_car_or_motorcycle = False

            for result in results:
                for bbox in result.boxes:
                    confidence = bbox.conf[0]
                    label = self.vehicle_model.names[int(bbox.cls)]

                    if label.lower() in ['car', 'motorcycle'] and confidence >= 0.75:
                        detected_car_or_motorcycle = True

                        license_plate_results = self.license_plate_model(frame)
                        detections = license_plate_results[0].boxes

                        for box in detections:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            license_plate_img = frame[y1:y2, x1:x2]
                            gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

                            results = reader.readtext(gray_plate, detail=1)
                            for res in results:
                                text = res[1]
                                ocr_confidence = res[2]
                                filtered_text = re.sub(r'[^A-Z0-9]', '', text.strip())

                                if ocr_confidence > 0.8 and 6 <= len(filtered_text) <= 7:
                                    self.save_image(frame, f"{filtered_text}.png")
                                    pause_detection = True
                                    print(
                                        f"Detected {label} with confidence {confidence:.2f} AND OCR detected text: {filtered_text} with confidence {ocr_confidence:.2f}")
                                    self.text_detected.emit(filtered_text)  # 使用 self.text_detected.emit
                                cv2.putText(frame, filtered_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

            cv2.imshow('YOLO Webcam Detection with License Plate Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)

    # 創建 VideoCaptureThread 的實例
    video_thread = VideoCaptureThread()

    # 連接信號到槽函數
    def update_label(text):
        ui.label.setText(text)

    video_thread.text_detected.connect(update_label)

    # 啟動線程
    video_thread.start()

    Dialog.show()
    sys.exit(app.exec())