# models.py
from database import db


# 連線資料庫的table
# 色盲點圖的題目圖片
class pic(db.Model):
    __tablename__='color_blind_question_pic'
    id=db.Column(db.Integer,primary_key=True)
    image_data=db.Column(db.String(150))

    def __init__(self, image_data):
        self.image_data = image_data
# 色盲點圖的答案圖片
class ans(db.Model):
    __tablename__='color_blind_ans_pic'
    id_ans_cb = db.Column(db.Integer, primary_key=True)
    image_data = db.Column(db.LargeBinary)

    def __init__(self, image_data=None):
        self.image_data = image_data

# 使用者作答的圖片
class user_ans(db.Model):
    __tablename__ = 'color_blind_user_ans_pic'
    
    id = db.Column(db.Integer, primary_key=True)  # 使用者作答題目的編號
    user_id = db.Column(db.String, primary_key=True)  # 使用者的 UUID
    question_id = db.Column(db.Integer)  # 紀錄使用者每一題題目是哪張
    image_data = db.Column(db.LargeBinary)  # 使用者作答圖片的 data
    
    __table_args__ = (
        db.PrimaryKeyConstraint('id', 'user_id'),  # 定義複合主鍵
    )

    def __init__(self, id=None, image_data=None, user_id=None, question_id=None):
        self.id = id
        self.user_id = user_id
        self.question_id = question_id
        self.image_data = image_data



