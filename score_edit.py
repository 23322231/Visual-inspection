# 連接的版本
from PIL import Image
import cv2
import numpy as np
from flask import Flask
width = 12  #可容許誤差寬度

def rgba_to_rgb(image):
    # 假設圖像帶有 alpha 通道，將其轉換為 RGB
    background_color = (255, 255, 255)
    new_img = np.full((image.shape[0], image.shape[1], 3), background_color, dtype=np.uint8)
    if image.shape[2] == 4:  # 檢查是否有 alpha 通道
        alpha = image[:, :, 3]
        for c in range(3):
            new_img[:, :, c] = (1.0 - alpha / 255.0) * background_color[c] + (alpha / 255.0) * image[:, :, c]
        return new_img
    return image[:, :, :3]  # 如果沒有 alpha 通道，直接返回 RGB

def Score_calculation(image_ans, image_user):
    # image_ans = cv2.cvtColor(image_ans, cv2.COLOR_BGR2GRAY)
    # image_user = rgba_to_rgb(image_user)
    # image_user = cv2.cvtColor(image_user, cv2.COLOR_BGR2GRAY)

    if len(image_ans.shape) == 3 and image_ans.shape[2] == 3:  # 3 通道表示 BGR
        image_ans = cv2.cvtColor(image_ans, cv2.COLOR_BGR2GRAY)

    # 如果使用者圖片不是灰度圖，則轉換為灰度圖
    if len(image_user.shape) == 3 and image_user.shape[2] == 4:  # 4 通道表示 RGBA
        image_user = rgba_to_rgb(image_user)
    if len(image_user.shape) == 3 and image_user.shape[2] == 3:  # 3 通道表示 RGB
        image_user = cv2.cvtColor(image_user, cv2.COLOR_BGR2GRAY)
    
    # 進行二值化處理
    _, image_ans = cv2.threshold(image_ans, 127, 255, cv2.THRESH_BINARY)
    _, image_user = cv2.threshold(image_user, 127, 255, cv2.THRESH_BINARY)

    # 進行 XOR 操作 (找出沒有與答案吻合的部分)
    xor_result = np.bitwise_xor(image_ans, image_user)

    # 與答案進行 AND 操作 (只留下吻合的黑色軌跡)
    and_result = np.bitwise_or(image_ans, xor_result)

    # 找出黑色像素 (與答案吻合的軌跡) 計算數量
    black_pixels = np.where(and_result == 0)
    black_pixel_count = len(black_pixels[0])

    # 找出答案黑色像素 (答案軌跡) 計算數量
    black_pixels_ans = np.where(image_ans == 0)
    black_pixel_count_ans = len(black_pixels_ans[0])

    # 計算吻合度 (與答案吻合的軌跡 / 答案軌跡)
    similarity = black_pixel_count / black_pixel_count_ans

    # 計算與最近吻合軌跡的距離
    distance_transform = cv2.distanceTransform(and_result, cv2.DIST_L2, 3)

    # 找出所有不吻合像素
    unmatched_pixel = np.where(xor_result == 255)
    score = 100  # 滿分 100

    if similarity < 0.4:  # 吻合度 < 0.4，直接設為 0 分 (色盲)
        score = 0

    for y, x in zip(unmatched_pixel[0], unmatched_pixel[1]):
        unmatched = distance_transform[y, x]
        if width < unmatched <= width * 3:  # 超過容許寬度 1-3 倍，扣 0.03 分
            score -= 0.03
        elif unmatched > width * 3:  # 超過容許寬度 3 倍以上，扣 0.08 分
            score -= 0.08

    return max(score, 0)  # 分數最低為 0 分

# def get_image_from_db(user_id, question_id):
#     from app import user_ans, ans # 在函数内部导入，避免循环导入
#     from app import db # 從主應用程序導入已初始化的 db
#     # 假設這裡已經有連接到資料庫，並且有 ORM 類別 user_ans
#     user_answer = db.session.query(user_ans).filter_by(user_id=user_id, question_id=question_id).first()
#     answer_image = db.session.query(ans).filter_by(id_ans_cb=question_id).first()

#     if not user_answer or not answer_image:
#         return None, None
    
#     # 從資料庫中提取二進制數據
#     image_user_data = user_answer.image_data
#     image_ans_data = answer_image.image_data

#     # 將二進制數據轉換為 NumPy 數組
#     image_user_array = np.frombuffer(image_user_data, np.uint8)
#     image_ans_array = np.frombuffer(image_ans_data, np.uint8)

#     # 使用 OpenCV 解碼為圖片
#     image_user = cv2.imdecode(image_user_array, cv2.IMREAD_UNCHANGED)
#     image_ans = cv2.imdecode(image_ans_array, cv2.IMREAD_UNCHANGED)

#     return image_ans, image_user
