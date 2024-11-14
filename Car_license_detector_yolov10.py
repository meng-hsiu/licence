# PyQt6 generated code for license.ui
from PyQt6 import QtWidgets, uic
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
import configparser


class MainApp(QtWidgets.QDialog):
    def __init__(self):
        super(MainApp, self).__init__()
        uic.loadUi("license.ui", self)  # 動態載入 UI

        # 設定視窗標題和 UI 元件
        self.setWindowTitle("車牌辨識")
        self.label = self.findChild(QtWidgets.QLabel, "label")
        self.label.setText("正在等待偵測...")

        # 初始化並連接影片擷取執行緒
        self.video_thread = VideoCaptureThread()
        self.video_thread.text_detected.connect(self.label.setText)
        self.video_thread.exit_signal.connect(self.close)
        self.video_thread.start()  # 啟動執行緒


class VideoCaptureThread(QThread):
    # 在 PyQt 中定義一個信號,用在 Qt 的事件處理系統中傳遞訊息或事件, 簡單來說就是我拿來傳遞上面的文字啦..
    text_detected = pyqtSignal(str)
    exit_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.last_detected_text = None
        self.pause_detection = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load models, 開發模式跟包裝後的model載入路徑是不同的
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
        self.vehicle_model = YOLO(os.path.join(base_path, 'yolov10n.pt'),verbose=False).to(self.device)
        self.license_plate_model = YOLO(os.path.join(base_path, 'license_plate_detector.pt'),verbose=False).to(self.device)

    # 儲存照片的function, 用@staticmethod可以不用實體化就拿來用
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

    # webcam運行
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

            # cv2 沒辦法直接打打中文...
            cv2.imshow('YOLO Webcam Detection with License Plate Recognition', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        # 按q退出後要釋放cv2和webcam的資源
        cap.release()
        cv2.destroyAllWindows()
        self.exit_signal.emit()
        sys.exit(1)

    # 判斷是否是車子或摩托車
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

    # 這段是在進行OCR的判斷, 也確認有沒有符合我設定的正規化條件
    def process_ocr_results(self, results, frame, x1, y1):
        for res in results:
            text, confidence = res[1], res[2]
            filtered_text = re.sub(r'[^A-Z0-9]', '', text.strip())
            cv2.putText(frame, filtered_text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if confidence > 0.8 and 6 <= len(filtered_text) <= 7 and filtered_text != self.last_detected_text:
                self.last_detected_text = filtered_text
                # cv2.imshow("Detected the car license", frame)
                print(f"Detected {filtered_text} with confidence {confidence:.2f}")
                try:
                    self.handle_database_interaction(filtered_text, frame)
                except Exception as e:
                    print(e)
                self.pause_detection = True
                return True
        return False

    # 建立資料庫連線與建立各種CRU
    def handle_database_interaction(self, plate_text, frame):
        # 資料庫連線字串設定, 改這樣比較不會推到github上導致看到帳密
        # 讀取配置檔案
        config = configparser.ConfigParser()
        config.read('config.ini')
        # 從配置檔案中獲取連線字串
        connection_string = (
            f"Driver={{{config['database']['driver']}}};"
            f"Server={config['database']['server']};"
            f"Database={config['database']['database']};"
            f"Uid={config['database']['uid']};"
            f"Pwd={config['database']['pwd']};"
            f"Encrypt={config['database']['encrypt']};"
            f"TrustServerCertificate={config['database']['trustServerCertificate']};"
            f"Connection Timeout={config['database']['connectionTimeout']};"
        )
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        time_now = datetime.now()

        # 檢查出入管理內有沒有已經存在了, 已經存在的話就是出場
        query_start = "SELECT * FROM EntryExitManagement WHERE license_plate_photo Like ? AND exit_time IS NULL;"
        cursor.execute(query_start, f"{plate_text}_%.png")
        row = cursor.fetchone()

        # columns = [column[0] for column in cursor.description]
        # print("欄位名稱:", columns)

        if row:
            # 已經存在的話
            entryexit_id = row.entryexit_id
            is_payment = row.payment_status
            car_id = row.car_id
            valid_time = row.valid_time
            parktype = row.parktype
            if is_payment:
                # 已經存在, 且已經繳費(如果是月租的話, 我在Insert的時候是將繳費狀態設成已繳費), 要檢查是不是月租
                # 月租檢查完後會銜接檢查有沒有超過最後離場時間的function
                try:
                    self.check_is_monthly(cursor, plate_text, time_now, frame,
                                          entryexit_id, car_id, valid_time, parktype)
                except Exception as ex:
                    print(ex)
                    self.text_detected.emit("出現了繳費問題, 請洽管理員")
            else:
                # 已經存在, 但沒繳錢
                self.text_detected.emit("尚未完成付款動作")
        else:
            # 沒有存在的話
            # 接下來進入檢查有沒有月租或預定的function
            self.check_parking_status(cursor, plate_text, time_now, frame)
        # 關閉資料庫的連線
        cursor.close()
        conn.close()

    # 離場判斷是不是月租的function
    def check_is_monthly(self, cursor, plate_text, time_now, frame, entryexit_id, car_id, valid_time, parktype):
        if parktype == "MonthlyRental":
            # 如果是月租, 更新出場時間, 完成, 付款費用, 付款時間, 預計最後離場時間
            cursor.execute("UPDATE EntryExitManagement "
                           "SET exit_time = ?, is_finish= ?, amount=?, "
                           "payment_time=?, valid_time=?, license_plate_keyin_time=? "
                           "WHERE entryexit_id = ?;",
                           time_now, 1, 0, time_now, time_now, time_now, entryexit_id)
            cursor.connection.commit()
            self.text_detected.emit("謝謝光臨")
        else:
            # 不是月租的話進入檢查預計最後離場時間的function
            try:
                self.check_valid_time(cursor, plate_text, time_now, frame, entryexit_id, car_id, valid_time)
            except Exception as error:
                print(error)

    # 判斷有沒有超過最後離場時間的function
    def check_valid_time(self, cursor, plate_text, time_now, frame, entryexit_id, car_id, valid_time):
        if valid_time > time_now:
            # 如果沒有超過出場時間
            cursor.execute("UPDATE EntryExitManagement SET exit_time = ?, is_finish= ? WHERE entryexit_id = ?;",
                           time_now, 1, entryexit_id)
            cursor.execute(
                "UPDATE ParkingLots SET validSpace = validSpace -1 WHERE lot_id = 1;"
            )
            cursor.connection.commit()
            self.text_detected.emit("謝謝光臨")
        else:
            # 如果超過了出場時間, insert一筆新的紀錄, 而這筆的開始時間是他的預計最後離場時間
            cursor.execute("UPDATE EntryExitManagement SET exit_time = ?, is_finish= ? WHERE entryexit_id = ?;",
                           time_now, 1, entryexit_id)
            cursor.connection.commit()
            self.insert_entry(cursor, plate_text, valid_time,"Reservation", frame, car_id)
            self.text_detected.emit("已超過出場時間, 請再繳費一次")


    # 判斷停車類型的function, 例如有沒有月租, 預定, 如果要加臨停判斷就是加在這的最後一個else那邊
    def check_parking_status(self, cursor, plate_text, time_now, frame):
        # 確認是否有月租 篩選條件, 停車場編號, 車牌編號, 有沒有付錢, 車子是在啟用狀態嗎, 如果有同時符合, 選最新的那筆
        # 選擇最新的那筆是因為可能在這個停車場月租兩次過了, 要以新的合約的end_date來判斷能不能入場
        query_monthly = ("SELECT * FROM MonthlyRental JOIN Car ON Car.car_id = MonthlyRental.car_id "
                         "WHERE lot_id = ? AND license_plate = ? AND payment_status = ? AND is_active = ? "
                         "ORDER BY end_date DESC;")
        # 停車場編號目前都先讓他預設是1
        cursor.execute(query_monthly, 1, plate_text, 1, 1)
        row = cursor.fetchone()

        if row and row.end_date > time_now:
            self.insert_entry(cursor, plate_text, time_now, "MonthlyRental", frame, row.car_id)
            self.text_detected.emit(f"歡迎光臨, {plate_text}")
        elif row:
            self.text_detected.emit("合約過期，或是尚未繳費")
        else:
            # 查詢是否有預定, lot_id預設1, 車牌號碼, 是不是完成狀態, 有沒有繳錢, 車子是不是啟用中, 同時滿足選最舊的那筆
            # 選擇最舊的那筆意味著還沒超時, 但他同時有預定兩筆, 如果超時is_finish就是1而不會是0了
            query_reservation = ("SELECT * FROM Reservation JOIN Car ON Car.car_id = Reservation.car_id "
                                 "WHERE lot_id = ? AND license_plate = ? AND is_finish = ? "
                                 "AND payment_status = ? AND is_active = ? "
                                 "ORDER BY valid_until;")
            cursor.execute(query_reservation, 1, plate_text, 0, 1, 1)
            row = cursor.fetchone()

            if row and not row.is_overdue:
                # 如果資料存在, 而且不是超時
                # 將Reservation的is_finish更新成完成
                cursor.execute("UPDATE Reservation SET is_finish = ? WHERE res_id = ?", 1, row.res_id)
                # 進入到新增一筆資料到出入管理的function
                self.insert_entry(cursor, plate_text, time_now, "Reservation", frame, row.car_id)
                self.text_detected.emit(f"歡迎光臨, {plate_text}")
            elif row:
                # 其實我在想這邊有需要嗎? 畢竟超時是由計時器去判斷, 如果超時is_finish應該也會變成1才對, 所以我在上面的查詢應該就擋掉了
                cursor.execute("UPDATE Reservation SET is_finish = 1, is_overdue = 1 WHERE res_id = ?", row.res_id)
                self.text_detected.emit("已逾時,請重新預定")
            else:
                self.text_detected.emit("沒有預定或月租,請重新確認")

    # 新增出入管理資料的function
    def insert_entry(self, cursor, plate_text, time_now, parktype, frame, car_id):
        # 先將time_now轉成字串格式, 不然檔案名字沒辦法用DateTime格式
        str_time_now = time_now.strftime("%Y%m%d_%H%M%S")

        # 判斷是不是月租, 如果是月租那他就是已經付款狀態了
        if parktype == "MonthlyRental":
            cursor.execute(
                "INSERT INTO "
                "EntryExitManagement (lot_id, car_id, parktype, license_plate_photo, entry_time, payment_status) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                1, car_id, parktype, f"{plate_text}_{str_time_now}.png", time_now, 1)
            # 儲存照片, 格式是車牌號碼_時間.png
            self.save_image(frame, f"{plate_text}_{str_time_now}.png")
            cursor.connection.commit()
        else:
            # 停車場預設是1, car_id是當我判斷完月租或預定資料表或超時但沒出場的出入管理時傳入, parktype同上, 照片檔案名字, 現在的時間(超時那邊是傳入Valid_time)
            cursor.execute(
                "INSERT INTO "
                "EntryExitManagement (lot_id, car_id, parktype, license_plate_photo, entry_time, payment_status) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                1, car_id, parktype, f"{plate_text}_{str_time_now}.png", time_now, 0)
            self.save_image(frame, f"{plate_text}_{str_time_now}.png")
            cursor.connection.commit()
            # 這是我拿來看偵測結果的frame用的
            # cv2.imshow("testtest", frame)
            # print("資料型態:", frame.dtype)
            # print("物件類型:", type(frame))
            # print("影像維度:", frame.shape)
            # print("plate_text"+plate_text)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
