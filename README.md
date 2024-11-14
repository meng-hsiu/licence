# 車牌辨識系統

## 專案的功能與目的
辨識車牌後, 將其串聯到小組的資料庫中做資料的比對, 比對完成後決定是否可以進出場

## 安裝環境
1. CUDA版本: 12.6
2. CUDNN: 9.3
3. 其他python使用的庫都包含在requirements.txt中

## 本專案的流程
1. 先使用yolov10來進行檢測, 如果是車子或摩托車才進行下一步
2. 再使用license_plate_detector辨識出車牌的部分
3. 切割出車牌部分後, 通過EasyOCR去辨識車牌內容
4. 進入正規化判斷(以台灣車牌為例, 6或7碼的英文+數字)
5. 串接到資料庫中進行CRU

## <font color=yellow;>注意事項</font>
該專案針對小組專題所製作, 所以會連到MSSQL, 因此沒有相應資料庫的人是沒辦法使用此專案的
<br>
使用前要先自己創立一個config.ini
內容大致如下
```commandline
[database]
driver = ODBC Driver 17 for SQL Server
server = SERVER的位置
database = 資料庫名字
uid = SQL帳號
pwd = SQL密碼
encrypt = yes
trustServerCertificate = no
connectionTimeout = 30
```