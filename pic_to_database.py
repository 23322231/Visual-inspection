from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, LargeBinary
from PIL import Image
import io

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:01057126@localhost/visual"
db = SQLAlchemy(app)

# class ImageModel(db.Model):
#     __tablename__ = 'color_blind_question_pic'
#     id = Column(Integer, primary_key=True)
#     image_data = Column(LargeBinary, nullable=False)
#     answer = Column(Integer, nullable=False)

# class ImageModel(db.Model):
#     __tablename__ = 'color_blind_ans_pic'
#     id_ans_cb = Column(Integer, primary_key=True)
#     image_data = Column(LargeBinary, nullable=False)
#     answer = Column(Integer, nullable=False)

class ImageModel(db.Model):
    __tablename__ = 'e_chart'
    id = Column(Integer, primary_key=True)
    chart = Column(LargeBinary, nullable=False)



def save_images_to_db(ids, image_paths):
    # if len(image_paths) != len(answers) or len(ids) != len(answers):
    #     raise ValueError("圖片數量和答案數量不匹配")

    images = []
    
    for image_id, image_path in zip(ids, image_paths):
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format)
            img_byte_arr = img_byte_arr.getvalue()

        # 創建一個新的ImageModel實例，並將其添加到列表中
        # new_image = ImageModel(id_ans_cb=image_id, image_data=img_byte_arr, answer=answer)
        # new_image = ImageModel(id=image_id, image_data=img_byte_arr, answer=answer)
        new_image = ImageModel(id=image_id, chart=img_byte_arr)
        images.append(new_image)
    
    # 一次性將所有圖片批量插入到資料庫中
    db.session.bulk_save_objects(images)
    db.session.commit()
    print(f"{len(images)} images and answers saved to database successfully.")

if __name__ == "__main__":
    id = [i for i in range(1, 31)]

    # image_paths = [
    #     "ans/2ans.jpg",
    #     "ans/3ans.jpg",
    #     "ans/5ans.jpg",
    #     "ans/6ans.jpg",
    #     "ans/7ans.jpg",
    #     "ans/8ans.jpg",
    #     "ans/12ans.jpg",
    #     "ans/15ans.jpg",
    #     "ans/16ans.jpg",
    #     "ans/26ans.jpg",
    #     "ans/29ans.jpg",
    #     "ans/35ans.jpg",
    #     "ans/42ans.jpg",
    #     "ans/45ans.jpg",
    #     "ans/57ans.jpg",
    #     "ans/73ans.jpg",
    #     "ans/74ans.jpg",
    #     "ans/96ans.jpg",
    #     "ans/97ans.jpg",
    #     "ans/100ans.jpg",
    #     "ans/101ans.jpg",
    #     "ans/102ans.jpg",
    #     "ans/103ans.jpg",
    #     "ans/104ans.jpg",
    #     "ans/105ans.jpg",
    #     "ans/106ans.jpg",
    #     "ans/107ans.jpg",
    #     "ans/108ans.jpg",
    #     "ans/109ans.jpg",
    #     "ans/110ans.jpg",
    # ]
    # image_paths = [
    #     "static/colorblind_image/2.jpg",
    #     "static/colorblind_image/3.jpg",
    #     "static/colorblind_image/5.jpg",
    #     "static/colorblind_image/6.jpg",
    #     "static/colorblind_image/7.jpg",
    #     "static/colorblind_image/8.jpg",
    #     "static/colorblind_image/12.jpg",
    #     "static/colorblind_image/15.jpg",
    #     "static/colorblind_image/16.jpg",
    #     "static/colorblind_image/26.jpg",
    #     "static/colorblind_image/29.jpg",
    #     "static/colorblind_image/35.jpg",
    #     "static/colorblind_image/42.jpg",
    #     "static/colorblind_image/45.jpg",
    #     "static/colorblind_image/57.jpg",
    #     "static/colorblind_image/73.jpg",
    #     "static/colorblind_image/74.jpg",
    #     "static/colorblind_image/96.jpg",
    #     "static/colorblind_image/97.jpg",
    #     "static/colorblind_image/100.jpg",
    #     "static/colorblind_image/101.jpg",
    #     "static/colorblind_image/102.jpg",
    #     "static/colorblind_image/103.jpg",
    #     "static/colorblind_image/104.jpg",
    #     "static/colorblind_image/105.jpg",
    #     "static/colorblind_image/106.jpg",
    #     "static/colorblind_image/107.jpg",
    #     "static/colorblind_image/108.jpg",
    #     "static/colorblind_image/109.jpg",
    #     "static/colorblind_image/110.jpg",
    # ]

    image_paths = [
        "static/e_chart/E_up.jpg",
        "static/e_chart/E_down.jpg",
        "static/e_chart/E_left.jpg",
        "static/e_chart/E_right.jpg"
    ]
    # answers = [2,3,5,6,7,8,12,15,16,26,29,35,42,45,57,73,74,96,97,100,101,102,103,104,105,106,107,108,109,110 ]  # 對應的答案列表 #不是數字的填100

    with app.app_context():
        db.create_all()  # 創建表
        save_images_to_db(id,image_paths)
