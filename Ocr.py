import cv2
import PIL
from PIL import Image
import numpy as np
import pytesseract
from ultralytics import YOLO



def area(row, col):
    global nn
    if bg[row][col] != 255:
        return
    bg[row][col] = lifearea  # 記錄生命區的編號
    if col > 1:  # 左方
        if bg[row][col - 1] == 255:
            nn += 1
            area(row, col - 1)
    if col < w - 1:  # 右方
        if bg[row][col + 1] == 255:
            nn += 1
            area(row, col + 1)
    if row > 1:  # 上方
        if bg[row - 1][col] == 255:
            nn += 1
            area(row - 1, col)
    if row < h - 1:  # 下方
        if bg[row + 1][col] == 255:
            nn += 1
            area(row + 1, col)


# Load the YOLO model
model = YOLO("license_plate_detector.pt")

# Load the image using OpenCV
img_read = cv2.imread("image.jpg")

# Convert the OpenCV image (NumPy array) to a PIL image
img_pil = Image.fromarray(cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB))

# Resize the image using PIL
img_pil = img_pil.resize((300, 225), Image.LANCZOS)

# Convert the resized PIL image back to a NumPy array for YOLO model processing
img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 將圖像轉換為灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 定義對比度增強的係數
alpha = 2.5  # 對比度增強係數，越大越顯著
beta = 0     # 亮度增強偏移量，這裡設為0表示只調整對比度

# 對比度調整
contrast_adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# 設定閾值，所有高於此值的像素將變為白色
threshold_value = 110  # 根據需要調整閾值

# 創建一個二進制掩碼，將高於閾值的像素設置為白色
_, binary_mask = cv2.threshold(contrast_adjusted, threshold_value, 255, cv2.THRESH_BINARY)

# 使用掩碼將高亮度像素轉換為白色
img[binary_mask == 255] = [255, 255, 255]  # 將高於閾值的像素設置為白色

# Perform object detection on the resized image
results = model(img)


detections = results[0].boxes

for box in detections:
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # bounding box coordinates

    # Crop the detected license plate area from the image
    license_plate_img = img[y1:y2, x1:x2]

    # Convert the cropped NumPy array to a PIL image
    img_pil2 = Image.fromarray(cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2RGB))

    # Resize the image to 140x40
    image2 = img_pil2.resize((140, 40), Image.LANCZOS)

    # Convert the resized PIL image back to a NumPy array
    image = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰階

    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # 轉為黑白
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    contours1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 尋找輪廓
    contours = contours1[0]  # 取得輪廓

    print(f"Number of contours found: {len(contours)}")
    if len(contours) == 0:
        print("No contours found.")

    letter_image_regions = []  # 文字圖形串列
    for contour in contours:  # 依序處理輪廓
        (x, y, w, h) = cv2.boundingRect(contour)  # 單一輪廓資料
        print(f"Contour found with x={x}, y={y}, w={w}, h={h}")
        letter_image_regions.append((x, y, w, h))  # 輪廓資料加入串列
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])  # 按X坐標排序
    print(letter_image_regions)
    # 先計算可以擷取的字元數
    count = 0  # 計算共擷取多少個字元

    for box in letter_image_regions:  # 依序處理輪廓資料
        x, y, w, h = box
        # x 必須介於 2~125 且寬度在 5~26、高度在 20~39 才是文字
        cv2.rectangle(license_plate_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if x >= 2 and x <= 125 and w >= 5 and w <= 26 and h >= 20 and h < 40:
            count += 1
    print(f"可以擷取的字元:{count}")
    if count < 6:  # 若字元數不足，可能是有兩個字元連在一起，將字元寬度放寬再重新擷取
        wmax = 35
    else:
        wmax = 26  # 正常字元寬度

    nChar = 0  # 計算共擷取多少個字元
    letterlist = []  # 儲存擷取的字元坐標
    for box in letter_image_regions:  # 依序處理輪廓資料
        x, y, w, h = box
        # x 必須介於 2~125 且寬度在 5~wmax、高度在 20~39 才是文字
        if x >= 2 and x <= 125 and w >= 5 and w <= wmax and h >= 20 and h < 40:
            nChar += 1
            letterlist.append((x, y, w, h))  # 儲存擷取的字元
    print(f"nChar可以擷取的字元:{nChar}")
    # 去除雜點
    for i in range(len(thresh)):  # i為高度
        for j in range(len(thresh[i])):  # j為寬度
            if thresh[i][j] == 255:  # 顏色為白色
                count = 0
                for k in range(-2, 3):
                    for l in range(-2, 3):
                        try:
                            if thresh[i + k][j + l] == 255:  # 若是白點就將count加1
                                count += 1
                        except IndexError:
                            pass
                if count <= 6:  # 週圍少於等於6個白點
                    thresh[i][j] = 0  # 將白點去除

    real_shape = []
    for i, box in enumerate(letterlist):  # 依序擷取所有的字元
        x, y, w, h = box
        bg = thresh[y:y + h, x:x + w]

        # 去除崎鄰地
        if i == 0 or i == nChar:  # 只去除第一字元和最後字元的崎鄰地
            lifearea = 0  # 生命區塊
            nn = 0  # 每個生命區塊的生命數
            life = []  # 記錄每個生命區塊的生命數串列
            for row in range(0, h):
                for col in range(0, w):
                    if bg[row][col] == 255:
                        nn = 1  # 生命起源
                        lifearea = lifearea + 1  # 生命區塊數
                        area(row, col)  # 以生命起源為起點探索每個生命區塊的總生命數
                        life.append(nn)

            maxlife = max(life)  # 找到最大的生命數
            indexmaxlife = life.index(maxlife)  # 找到最大的生命數的區塊編號

            for row in range(0, h):
                for col in range(0, w):
                    if bg[row][col] == indexmaxlife + 1:
                        bg[row][col] = 255
                    else:
                        bg[row][col] = 0

        real_shape.append(bg)  # 加入字元

    # 在圖片週圍加白色空白OCR才能辨識
    newH, newW = thresh.shape

    space = 8  # 空白寬度
    offset = 2
    bg = np.zeros((newH + space * 2, newW + space * 2 + nChar * 3, 1), np.uint8)  # 建立背景
    bg.fill(0)  # 背景黑色

    if not real_shape:
        print("real_shape is empty.")
    else:
        print(f"real_shape contains {len(real_shape)} elements.")

    # 將車牌文字加入黑色背景圖片中
    for i, letter in enumerate(real_shape):
        if letter.size == 0:  # 檢查字元圖像是否為空
            print(f'letter_{i} is empty and will not be displayed.')
            continue
        cv2.imshow(f'letter_{i}', letter)
        cv2.waitKey(0)
        h = letter.shape[0]  # 原來文字圖形的高、寬
        w = letter.shape[1]
        x = letterlist[i][0]  # 原來文字圖形的位置
        y = letterlist[i][1]
        for row in range(h):  # 將文字圖片加入背景
            for col in range(w):
                bg[space + y + row][space + x + col + i * offset] = letter[row][col]  # 擷取圖形

    _, bg = cv2.threshold(bg, 128, 255, cv2.THRESH_BINARY_INV)  # 轉為白色背景、黑色文字

    cv2.imshow('Contrast Adjusted Image', contrast_adjusted)
    cv2.imshow('Thresholded Image', thresh)
    cv2.imshow('License Plate with Bounding Boxes', license_plate_img)
    cv2.imshow('cutimage', img)
    cv2.imshow('image', image)  # 顯示原始圖形
    cv2.imshow('bg', bg)  # 顯示組合的字元
    cv2.moveWindow("image", 500, 250)  # 將視窗移到指定位置
    cv2.moveWindow("bg", 500, 350)  # 將視窗移到指定位置

    text = pytesseract.image_to_string(bg, lang='eng', config='--psm 11')  # psm 8 for single word or line
    print(f"Detected license plate number: {text.strip()}")

    key = cv2.waitKey(0)  # 按任意鍵結束
    cv2.destroyAllWindows()




