# PyQt6 generated code for license.ui
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal
import cv2
from ultralytics import YOLO
import easyocr
import re
import os
import sys
import pyodbc
from datetime import datetime
import torch


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

    def __init__(self):
        super().__init__()
        self.last_detected_text = None
        self.pause_detection = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load models
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
        self.vehicle_model = YOLO(os.path.join(base_path, 'yolov10n.pt'),verbose=False).to(self.device)
        self.license_plate_model = YOLO(os.path.join(base_path, 'license_plate_detector.pt'),verbose=False).to(self.device)

    @staticmethod
    def save_image(image, filename):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
        path = os.path.join(base_path, "detected_plates", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: 沒辦法打開webcam")
            return

        reader = easyocr.Reader(['en'])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: 讀取不到影像")
                break

            if self.pause_detection:
                if cv2.waitKey(1) == ord('r'):
                    self.pause_detection = False
                    self.last_detected_text = None
                    self.text_detected.emit("正在等待偵測...")
                continue

            # Detect car or motorcycle
            if self.detect_vehicle(frame, reader):
                continue

            cv2.imshow('YOLO Webcam Detection with License Plate Recognition', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def detect_vehicle(self, frame, reader):
        results = self.vehicle_model(frame,verbose=False)
        for result in results:
            for bbox in result.boxes:
                confidence = bbox.conf[0]
                label = self.vehicle_model.names[int(bbox.cls)]

                if label.lower() in ['car', 'motorcycle'] and confidence >= 0.75:
                    license_plate_results = self.license_plate_model(frame,verbose=False)
                    for box in license_plate_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cropped_plate = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        results = reader.readtext(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY), detail=1)
                        if self.process_ocr_results(results, frame, x1, y1):
                            return True
        return False

    def process_ocr_results(self, results, frame, x1, y1):
        for res in results:
            text, confidence = res[1], res[2]
            filtered_text = re.sub(r'[^A-Z0-9]', '', text.strip())

            if confidence > 0.8 and 6 <= len(filtered_text) <= 7 and filtered_text != self.last_detected_text:
                self.last_detected_text = filtered_text

                cv2.putText(frame, filtered_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.pause_detection = True
                print(f"Detected {filtered_text} with confidence {confidence:.2f}")
                cv2.imshow("Detect frame of the car license", frame)
                self.handle_database_interaction(filtered_text, frame)
                return True
        return False

    def handle_database_interaction(self, plate_text, frame):
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=MyGoParking;Trusted_Connection=yes;")
        cursor = conn.cursor()
        time_now = datetime.now()

        # Check entry or exit status
        query_start = "SELECT * FROM EntryExitManagement WHERE license_plate_photo = ? AND exit_time IS NULL;"
        cursor.execute(query_start, plate_text + ".png")
        row = cursor.fetchone()

        if row:
            entryexit_id = row.entryexit_id
            is_payment = row.payment_status
            if is_payment:
                cursor.execute("UPDATE EntryExitManagement SET exit_time = ? WHERE entryexit_id = ?", time_now,
                               entryexit_id)
                conn.commit()
                self.text_detected.emit("謝謝光臨")
            else:
                self.text_detected.emit("尚未完成付款動作")
        else:
            # Check monthly rental and reservation statuses
            self.check_parking_status(cursor, plate_text, time_now, frame)
        cursor.close()
        conn.close()

    def check_parking_status(self, cursor, plate_text, time_now, frame):
        # 確認是否有月租 篩選條件, 停車場編號, 車牌編號, 有沒有付錢, 如果有同時符合選最新的那筆
        query_monthly = "SELECT * FROM MonthlyRental JOIN Car ON Car.car_id = MonthlyRental.car_id WHERE lot_id = ? AND license_plate = ? AND payment_status = 1 ORDER BY end_date DESC;"
        # 停車場編號目前都先讓他預設是1
        cursor.execute(query_monthly, 1, plate_text)
        row = cursor.fetchone()

        if row and row.end_date > time_now:
            self.insert_entry(cursor, plate_text, time_now, "MonthlyRental", frame)
            self.text_detected.emit(f"歡迎光臨, {plate_text}")
        elif row:
            self.text_detected.emit("合約過期，或是尚未繳費")
        else:
            # Check for reservation
            query_reservation = "SELECT * FROM Reservation JOIN Car ON Car.car_id = Reservation.car_id WHERE lot_id = ? AND license_plate = ? AND is_finish = 0 AND deposit_status = 1 ORDER BY valid_until;"
            cursor.execute(query_reservation, 1, plate_text)
            row = cursor.fetchone()

            if row and not row.is_overdue:
                cursor.execute("UPDATE Reservation SET is_finish = 1 WHERE res_id = ?", row.res_id)
                self.insert_entry(cursor, plate_text, time_now, "Reservation", frame)
                self.text_detected.emit(f"歡迎光臨, {plate_text}")
            elif row:
                cursor.execute("UPDATE Reservation SET is_finish = 1, is_overdue = 1 WHERE res_id = ?", row.res_id)
                self.text_detected.emit("已逾時,請重新預定")
            else:
                self.text_detected.emit("沒有預定或月租,請重新確認")

    def insert_entry(self, cursor, plate_text, time_now, parktype, frame):
        cursor.execute(
            "INSERT INTO EntryExitManagement (lot_id, car_id, parktype, license_plate_photo, entry_time) VALUES (?, ?, ?, ?, ?);",
            1, cursor.fetchone().car_id, parktype, plate_text + ".png", time_now)
        self.save_image(frame, f"{plate_text}_{time_now}.png")
        print("plate_text"+plate_text)
        cursor.connection.commit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)

    video_thread = VideoCaptureThread()
    video_thread.text_detected.connect(ui.label.setText)

    try:
        video_thread.start()
        Dialog.show()
        sys.exit(app.exec())
    except Exception as e:
        QtWidgets.QMessageBox.critical(Dialog, "Error", f"An error occurred: {str(e)}")
        sys.exit(1)
