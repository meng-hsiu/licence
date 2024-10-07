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
    # 加入判斷式 如果是打包後就用相對,如果是開發模式就絕對路徑
    if getattr(sys, 'frozen', False):
        # 如果是打包後的應用，使用 _MEIPASS
        base_path = sys._MEIPASS
    else:
        # 如果是開發模式，使用當前腳本目錄
        base_path = os.path.dirname(__file__)

    def __init__(self):
        super().__init__()
        self.last_detected_text = None  # 初始化最近偵測的文字
        self.pause_detection = False

    # 檢查是否有可用的 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vehicle_model_path = os.path.join(base_path, 'yolov10n.pt')
    license_plate_model_path = os.path.join(base_path, 'license_plate_detector.pt')

    # 加載車輛模型和車牌模型時指定設備
    vehicle_model = YOLO(vehicle_model_path).to(device)
    license_plate_model = YOLO(license_plate_model_path).to(device)

    @staticmethod
    def save_image(image, filename):
        if getattr(sys, 'frozen', False):
            # 如果是打包後的應用，使用 _MEIPASS
            base_path = sys._MEIPASS
        else:
            # 如果是開發模式，使用當前腳本目錄
            base_path = os.path.dirname(__file__)

        path = os.path.join(base_path, "detected_plates", filename)
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
                    self.last_detected_text = None
                    self.text_detected.emit("正在等待偵測...")
                continue

            results = self.vehicle_model(frame, device=self.device)
            detected_car_or_motorcycle = False

            for result in results:
                for bbox in result.boxes:
                    confidence = bbox.conf[0]
                    label = self.vehicle_model.names[int(bbox.cls)]

                    if label.lower() in ['car', 'motorcycle'] and confidence >= 0.75:
                        detected_car_or_motorcycle = True

                        # 車牌偵測時指定設備
                        license_plate_results = self.license_plate_model(frame, device=self.device)
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
                                    # 只有在新的車牌號碼不同於最近的結果時，才執行以下代碼
                                    if filtered_text != self.last_detected_text:
                                        self.last_detected_text = filtered_text  # 更新最近的偵測結果
                                        self.save_image(frame, f"{filtered_text}.png")
                                        pause_detection = True
                                        print(
                                            f"Detected {label} with confidence {confidence:.2f} AND OCR detected text: {filtered_text} with confidence {ocr_confidence:.2f}")

                                    # 連線到資料庫
                                    conn = pyodbc.connect(
                                        "DRIVER={ODBC Driver 17 for SQL Server};"
                                        "SERVER=localhost;"
                                        "DATABASE=MyGoParking;"
                                        "Trusted_Connection=yes;"
                                    )
                                    cursor = conn.cursor()
                                    # 先搜尋看看有沒有已經存在的車牌且Exit是空值,如果有,代表他已經進場了
                                    # 而且也不會搜尋到其他筆 因為同一個車牌號碼不會同時進場兩次, 也不會發生不同停車場有同一個車牌號碼, 所以肯定有一個是出場
                                    query_start = "SELECT * FROM EntryExitManagement where license_plate_photo = ? AND exit_time is Null ;"
                                    cursor.execute(query_start, filtered_text+".png")
                                    row = cursor.fetchone()
                                    # 紀錄當前時間 以用來確認月租或者是預訂有沒有超時 和 拿來寫入進出場時間
                                    time_now = datetime.now()
                                    # 如果是row是True代表他有在
                                    if row:
                                        is_payment = row.payment_status
                                        entryexit_id = row.entryexit_id
                                        if is_payment:
                                            query_Exit = "UPDATE EntryExitManagement SET exit_time = ? where entryexit_id = ?;"
                                            cursor.execute(query_Exit, time_now, entryexit_id)
                                            conn.commit()
                                            self.text_detected.emit("謝謝光臨")
                                            cursor.close()
                                            conn.close()
                                        else:
                                            self.text_detected.emit("尚未完成付款動作")
                                            cursor.close()
                                            conn.close()
                                    else:
                                        car_id = 0
                                        parktype = ""
                                        # 確認是否有月租 篩選條件, 停車場編號, 車牌編號, 有沒有付錢, 如果有同時符合選最新的那筆
                                        query = "SELECT * FROM MonthlyRental as M JOIN Car as C on C.car_id = M.car_id WHERE lot_id=? AND license_plate = ? AND M.payment_status = 1 ORDER BY M.end_date DESC;"
                                        # 預設我的停車都是前金 所以lot_id是1
                                        cursor.execute(query, 1, filtered_text)
                                        is_m = False
                                        is_overtime_m = False
                                        row = cursor.fetchone()
                                        if row:
                                            is_m = True
                                            car_id = row.car_id
                                            end_time = row.end_date
                                            if end_time > time_now:
                                                parktype = "MonthlyRental"
                                                query_insert_m = "INSERT EntryExitManagement (lot_id, car_id, parktype, license_plate_photo,entry_time) VALUES (?, ?, ?, ?, ?);"
                                                cursor.execute(query_insert_m, 1, car_id, parktype, filtered_text + ".png", time_now)
                                                conn.commit()
                                                cursor.close()
                                                conn.close()
                                                self.text_detected.emit(f"歡迎光臨,{filtered_text}")
                                            else:
                                                self.text_detected.emit("合約過期，或是尚未繳費")
                                                cursor.close()
                                                conn.close()
                                        # 確認是否有預定 而且要選擇還未完成的預訂訂單 篩選條件停車場編號, 車牌, 有沒有完成, 有沒有付錢, 如果有兩筆都符合這些條件挑最舊的那筆
                                        query2 = "SELECT * FROM Reservation as R JOIN Car as C on C.car_id = R.car_id WHERE lot_id=? AND license_plate=? AND R.is_finish=0 AND R.deposit_status = 1 ORDER BY valid_until;"
                                        cursor.execute(query2, 1, filtered_text)
                                        row = cursor.fetchone()
                                        # 用來儲存預定ID 因為要幫這筆改成finished
                                        res_id = 0
                                        is_r = False
                                        is_overtime_r = False
                                        if row:
                                            is_r = True
                                            car_id = row.car_id
                                            parktype = "Reservation"
                                            # 用來儲存如果他有預定的話 要對預訂進行更動
                                            res_id = row.res_id
                                            is_overdue = row.is_overdue
                                            car_id = row.car_id
                                            # 更動預定資料表 表示他已完成
                                            if not is_overdue:
                                                query_is_finished = "UPDATE Reservation SET is_finish = ? WHERE res_id = ?;"
                                                cursor.execute(query_is_finished, 1, res_id)
                                                conn.commit()
                                                # 新增資料在出入管理
                                                query_insert_r = "INSERT EntryExitManagement (lot_id, car_id, parktype, license_plate_photo,entry_time) VALUES (?, ?, ?, ?, ?); "
                                                cursor.execute(query_insert_r, 1, car_id, parktype, filtered_text+".png", time_now)
                                                # 我先預設一個出場時間是1800年 所以在進出場時會多一個判斷 如果我抓到出場時間是<1900 那我就會判斷他是進場中的車子
                                                conn.commit()
                                                self.text_detected.emit(f"歡迎光臨,{filtered_text}")
                                                cursor.close()
                                                conn.close()
                                            else:
                                                is_overtime_r = True
                                                query_overtime = "UPDATE Reservation SET is_finish = ?,is_overdue = ? WHERE res_id = ?;"
                                                cursor.execute(query_overtime, 1, 1, res_id)
                                                conn.commit()
                                                self.text_detected.emit("已逾時,請重新預定")
                                                cursor.close()
                                                conn.close()
                                        # 下面那行同理這個意思 if is_m == False and is_r == False: 只是不夠Pythonic
                                        if not is_m and not is_r:
                                            self.text_detected.emit("沒有預定或月租,請重新確認")
                                        elif is_overtime_r:
                                            self.text_detected.emit("預定時間已超時")




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