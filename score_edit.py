# 連接的版本
from PIL import Image
import cv2
import numpy as np
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, LargeBinary
from io import BytesIO

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://april0909:TevhMabtcGLrRlyn1rqrnfVcI5sVIKsH@dpg-cr1ljs5umphs73afhad0-a.oregon-postgres.render.com/data_0ol7"

db = SQLAlchemy(app)

class pic(db.Model):
    __tablename__ = 'color_blind_ans_pic'
    id = db.Column(db.Integer, primary_key=True)
    image_data = db.Column(db.LargeBinary)

    def __init__(self, image_data):
        self.image_data = image_data

class user(db.Model):
    __tablename__ = 'user_ans'
    id = db.Column(db.Integer, primary_key=True)
    image_data = db.Column(db.LargeBinary)

    def __init__(self, image_data):
        self.image_data = image_data

def rgba_to_rgb(image, background_color=(255, 255, 255)):
    if image.shape[2] == 4:  # Check if image has alpha channel
        # Create a white background image
        background = np.full((image.shape[0], image.shape[1], 3), background_color, dtype=np.uint8)
        # Separate alpha channel from the image
        alpha_channel = image[:, :, 3] / 255.0
        for c in range(3):
            image[:, :, c] = image[:, :, c] * alpha_channel + background[:, :, c] * (1 - alpha_channel)
        return image[:, :, :3]  # Return RGB image
    else:
        return image

def Score_calculation():
    width = 12  # Tolerance width
    # 从数据库中读取正确答案图片
    colorblind_test = db.session.query(pic).filter(pic.id == 2).first()
    if not colorblind_test:
        return "Answer image not found in the database."

    user_test = db.session.query(user).filter(user.id == 85).first()
    if not user_test:
        return "User image not found in the database."

    # 将二进制数据转换为 NumPy 数组
    img_array_ans = np.frombuffer(colorblind_test.image_data, dtype=np.uint8)
    img_array_user = np.frombuffer(user_test.image_data, dtype=np.uint8)

    # 使用 OpenCV 解码图片
    image_ans = cv2.imdecode(img_array_ans, cv2.IMREAD_UNCHANGED)
    image_user = cv2.imdecode(img_array_user, cv2.IMREAD_UNCHANGED)

    # 检查解码后的图像是否为空
    if image_ans is None:
        return "Failed to decode answer image."
    if image_user is None:
        return "Failed to decode user image."

    # 转换 RGBA 为 RGB（如果需要）
    image_ans = rgba_to_rgb(image_ans)
    image_user = rgba_to_rgb(image_user)

    # 显示从数据库中读取的图像
    cv2.imshow("Answer Image", image_ans)
    cv2.imshow("User Image", image_user)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 转换为灰度图像
    image_ans = cv2.cvtColor(image_ans, cv2.COLOR_BGR2GRAY)
    image_user = cv2.cvtColor(image_user, cv2.COLOR_BGR2GRAY)

    # 进行二值化处理
    _, image_ans = cv2.threshold(image_ans, 127, 255, cv2.THRESH_BINARY)
    _, image_user = cv2.threshold(image_user, 127, 255, cv2.THRESH_BINARY)

    # 进行XOR操作(找出没有与答案吻合的部分)
    xor_result = np.bitwise_xor(image_ans, image_user)
    # 与答案进行AND操作(只留下吻合的黑色轨迹,把做XOR改变的黑色背景变回白色,防止不吻合的部分与背景做 distance transform)
    and_result = np.bitwise_or(image_ans, xor_result)

    # 找出黑色像素(与答案吻合的轨迹)、计算数量
    black_pixels = np.where(and_result == 0)
    black_pixel_count = len(black_pixels[0])
    # 找出答案黑色像素(答案轨迹)、计算数量
    black_pixels_ans = np.where(image_ans == 0)
    black_pixel_count_ans = len(black_pixels_ans[0])

    # 检查答案图片中是否有黑色像素，避免除以零错误
    if black_pixel_count_ans == 0:
        return 0  # 或者其他适合的分数或错误处理

    # 计算吻合度(与答案吻合的轨迹/答案轨迹)
    similarity = black_pixel_count / black_pixel_count_ans

    # 计算与最近的吻合轨迹的距离
    distance_transform = cv2.distanceTransform(and_result, cv2.DIST_L2, 3)

    unmatched_pixel = np.where(xor_result == 255)  # 找出所有不吻合像素
    score = 100  # 满分100
    if similarity < 0.4:  # 吻合度<0.4,分数直接设为0分(色盲)
        score = 0

    for y, x in zip(unmatched_pixel[0], unmatched_pixel[1]):
        unmatched = distance_transform[y, x]
        if unmatched > width and unmatched <= width * 3:  # 超過容許寬度1-3倍,扣0.03分   
            score -= 0.03
        elif unmatched > width * 3:  # 超過容許寬度3倍以上,扣0.08分
            score -= 0.08

    if score < 0:  # 分數最低為0分
        score = 0

    return score

if __name__ == '__main__':
    with app.app_context():
        score = Score_calculation()
        print(f"Score: {score}")
