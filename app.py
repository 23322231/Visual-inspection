from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import base64
from PIL import Image
from io import BytesIO
import random
import uuid
from sqlalchemy.dialects.postgresql import UUID
import os
from sqlalchemy.exc import IntegrityError
from psycopg2 import Binary
import string
from flask import Response,send_from_directory
import psycopg2
import re
from hashlib import md5

# 定義候選的數字列表
numbers = [2, 3, 5, 6, 7, 8, 12, 15, 16, 26, 29, 35, 42, 45, 57, 73, 74, 96, 97, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:01057126@localhost/visual"

app.secret_key = "apri25805645l01057126===+++++"  # 用於會話加密的密鑰

# 連接到 PostgreSQL 資料庫
conn = psycopg2.connect(
    dbname="visual", 
    user="postgres", 
    password="01057126", 
    host="localhost", 
    port="5432"
)

db = SQLAlchemy(app)
socketio = SocketIO(app , ping_timeout=60, ping_interval=25,cors_allowed_origins="*") # cors_allowed_origins="*" 可以允許任何来源的跨域請求。

current_image = None

# 這裡是提供靜態檔案的路由，將內容類型指定為 JavaScript
@app.route('/static/assets/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/assets/js', filename, mimetype='text/javascript')

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

        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quiz')
def start():
    return render_template('quiz.html')

@app.route('/qrcode')
def qrcode():
    return render_template('qrcode.html')

@app.route('/open_pic')
def open_pic():
    return render_template('open_pic.html')

@app.route('/camera')
def choose():
    return render_template('camera_try.html')

@app.route('/myopia')
def myopia():
    return render_template('myopia.html')
    
#點圖製作功能頁面
@app.route('/ishihara-test')
def elements():
    return render_template('ishihara-test.html')

@app.route('/comfirm_colordot')
def comfirm_colordot():
    return render_template('comfirm_colordot.html')

@app.route('/finish')
def finish():
    return render_template('finish.html')

@app.route('/result')
def result():
    return render_template('result.html')

# 色盲點圖顯示題目圖片
@app.route('/next-image')
def next_image():
    print("執行了")
    random_id = random.randint(1, 30)
    session['random_id'] = random_id  # 将random_id存储到session中

    colorblind_test = db.session.query(pic).filter(pic.id == random_id).first()
    if colorblind_test:
        base64_data = base64.b64encode(colorblind_test.image_data).decode('utf-8')
        next_image_url = f"data:image/jpeg;base64,{base64_data}"
        return jsonify({'nextImageUrl': next_image_url})
    else:
        return jsonify({'error': 'No image found'}), 404 

@app.route('/result_cb', methods=['POST'])
def result_cb():
    data = request.get_json()  # 獲取前端傳來的 JSON 數據
    user_id = data.get('user_id')  # 從 JSON 中獲取 user_id
    index = data.get('index')  # 從 JSON 中獲取 index
    if not user_id:
        return jsonify({'error': 'User ID not provided'}), 400
    if not index:
        return jsonify({'error': 'Index not provided'}), 400
    
    print(f"Fetching images for user_id: {user_id}")
    images = []
    answer = user_ans.query.filter_by(user_id=user_id, id=index).first()
        
    if answer and answer.image_data:
        # 將二進位圖片數據轉換為 base64
        encoded_image = base64.b64encode(answer.image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{encoded_image}"  # 圖片格式為 JPEG
        print("--------------------")
        print(f"Image {index} encoded")
        images.append(image_url)
    else:
        images.append(None)
    
    return jsonify(images)  # 返回圖片的 JSON 列表



# 上傳使用者作答圖片
@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    user_id = session.get('user_id', None)

    if not user_id:
            print("找不到user_id")
            return jsonify({'error': 'User ID not found in session'}), 321
    
    if 'image' not in data:
        return jsonify({'error': 'No image data found'})
    
    image_data = data['image']
################################################################################

    # 處理了圖片資料量太大，資料庫無法處理的問題
    # 使用正則表達式處理不同圖片格式的前綴
    image_data = re.sub(r'^data:image/\w+;base64,', '', image_data)

    # 修正 base64 填充
    def fix_base64_padding(base64_string):
        # 根據長度缺少的部分，補足 `=` 符號
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += '=' * (4 - missing_padding)
        return base64_string

    image_data = fix_base64_padding(image_data)

    # image_data = image_data.replace('data:image/jpeg;base64,', '')
    # 解碼 base64 字符串

    try:
        binary_image_data = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({'error': 'Invalid base64 data', 'message': str(e)})
################################################################################
    # 获取 completedQuestions 數據
    completed_questions = data['completedQuestions']

    # 從session中獲取random_id
    random_id = session.get('random_id', None)
    print(random_id)
    if random_id is None:
        return jsonify({'error': 'Random ID not found'}), 111
    print(f"User ID: {user_id}, Random ID: {random_id}, Image Data Length: {len(binary_image_data)}")
    new_image = user_ans(id=completed_questions,image_data=binary_image_data,user_id=user_id,question_id=random_id)
    db.session.add(new_image)
    db.session.commit()

    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # 將 RGBA 圖像轉換為 RGB 格式
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image.save('uploaded_image.jpg')
    

    return jsonify({'message': 'Image uploaded successfully'})

# 由handwrite.html發送'img-connect'加上下一張題目的圖片的Base64編碼資料 
@socketio.on('img-connect')
def handle_connect(data):
    print("收到 img-connect 事件")
    current_image = data.get('background-image')
    if current_image:
        # print("發送圖片 URL:", current_image)  # 確認收到的圖片 URL
        emit('update_image', {'image': current_image}, broadcast=True)
        print("圖片發送成功!")
    else:
        print("錯誤: 沒有圖片 URL")
    
@socketio.on('finish_cb')
def finish_cb():
    emit('goto_result',broadcast=True)
    

# 確認進入色盲點圖頁面 傳送user uuid
@socketio.on('confirmDrawing')
def handle_confirm_drawing(data):
    print("Received confirmDrawing event")  # 確認事件觸發
    url_suffix = data.get('urlSuffix')
    if url_suffix:
        # print("=======")
        print(f'Received urlSuffix: {url_suffix}')
        emit('confirm', {'urlSuffix': url_suffix},broadcast=True)
    else:
        print('No URL suffix provided.')

@app.route('/handwrite')
def handwrite():
    user_uuid = request.args.get('session')  # 从查询参数中获取 session ID
    if user_uuid:
        return render_template('handwrite.html', user_uuid=user_uuid)
    else:
        return "User UUID not provided", 400

@app.route('/color_blind_spot_map')
def color_blind_spot_map():
    user_uuid = request.args.get('session')  # 从查詢参數中獲取 session ID
    if user_uuid:
        return render_template('/color_blind_spot_map.html', user_uuid=user_uuid)
    else:
        return "User UUID not provided", 400

# 產生唯一的網址 handwrite?session=
@app.route('/generate-url', methods=['GET'])
def generate_url():
    user_id = str(uuid.uuid4())  # 生成UUID
    unique_url = f"{request.host_url}handwrite?session={user_id}"
    session['user_id']=user_id
    session['unique_url'] = unique_url  # 存儲到會話中
    return jsonify({'url': unique_url})

@app.route('/generate-url-qrcode')
def generate_url_qrcode():
    session_id = str(uuid.uuid4())  # 生成唯一的sessionID
    unique_url = f"{request.host_url}comfirm_colordot?session={session_id}"
    return jsonify(url=unique_url)

@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('startSession')
def handle_start_session(data):
    session_id = data.get('sessionID')
    print(f"Session started: {session_id}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    # socketio.run(app,host='0.0.0.0', port=5000, debug=True)
