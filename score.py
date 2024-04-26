from PIL import Image
import cv2
import numpy as np

width = 12  #可容許誤差寬度

def rgba_to_rgb(image_path, background_color=(255, 255, 255)):
    # 讀取圖片
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # 建立一張白色底的圖片，與原圖尺寸相同
    new_img = np.full((img.shape[0], img.shape[1], 3), background_color, dtype=np.uint8)
    # 將原圖的 RGB 通道與 Alpha 通道合併到新圖上
    alpha = img[:, :, 3]
    for c in range(3):
        new_img[:, :, c] = (1.0 - alpha / 255.0) * background_color[c] + (alpha / 255.0) * img[:, :, c]

    return new_img


def Score_calculation(image_ans, image_user):
    image_ans = cv2.imread(image_ans)
    image_user = rgba_to_rgb(image_user)
    image_ans = cv2.cvtColor(image_ans, cv2.COLOR_BGR2GRAY)
    image_user = cv2.cvtColor(image_user, cv2.COLOR_BGR2GRAY)
    # 進行二值化處理
    _, image_ans = cv2.threshold(image_ans, 127, 255, cv2.THRESH_BINARY)
    _, image_user = cv2.threshold(image_user, 127, 255, cv2.THRESH_BINARY)
    # 進行XOR操作(找出沒有與答案吻合的部分)
    xor_result = np.bitwise_xor(image_ans, image_user)
    # cv2.imwrite("xor_result.png", xor_result)
    # 與答案進行AND操作(只留下吻合的黑色軌跡,把做XOR改變的黑色背景變回白色,防止不吻合的部分與背景做 distance transform)
    and_result = np.bitwise_or(image_ans, xor_result)
    # cv2.imwrite("and_result.png", and_result)
    # 找出黑色像素(與答案吻合的軌跡)、計算數量
    black_pixels = np.where(and_result == 0)
    black_pixel_count = len(black_pixels[0])
    # 找出答案黑色像素(答案軌跡)、計算數量
    black_pixels_ans = np.where(image_ans == 0)
    black_pixel_count_ans = len(black_pixels_ans[0])
    #計算吻合度(與答案吻合的軌跡/答案軌跡)
    similarity = black_pixel_count/black_pixel_count_ans

    # 計算與最近的吻合軌跡的距離
    distance_transform = cv2.distanceTransform(and_result, cv2.DIST_L2, 3)
    # cv2.imwrite("dt_result.png", distance_transform)

    unmatched_pixel = np.where(xor_result == 255)    #找出所有不吻合像素
    score = 100     #滿分100
    if(similarity < 0.4):  # 吻合度<0.4,分數直接設為0分(色盲)
        score = 0

    for y, x in zip(unmatched_pixel[0], unmatched_pixel[1]):
        unmatched = distance_transform[y, x]
        if unmatched > width and unmatched <= width * 3:    # 超過容許寬度1-3倍,扣0.03分   
            score -= 0.03
        elif unmatched > width * 3:    # 超過容許寬度3倍以上,扣0.08分
            score -= 0.08

    if score<0: # 分數最低為0分
        score = 0
    
    return score


print(Score_calculation("ans/1ans.png", "uploaded_image.png"))
