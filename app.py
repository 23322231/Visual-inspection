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
import subprocess
import json
import numpy as np
import cv2
from score_edit import Score_calculation
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import tensorflow as tf
import detect_face

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



global depth_value, remaining_time

# 定義候選的數字列表
numbers = [2, 3, 5, 6, 7, 8, 12, 15, 16, 26, 29, 35, 42, 45, 57, 73, 74, 96, 97, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:01057126@localhost/visual"
app.secret_key = "apri25805645l01057126===+++++"  # 用於會話加密的密鑰
# db = db.init_app(app)
db = SQLAlchemy(app)
# from database_model import pic, ans, user_ans  # 從 models.py 中導入模型
# 連接到 PostgreSQL 資料庫
conn = psycopg2.connect(
    dbname="visual", 
    user="postgres", 
    password="01057126", 
    host="localhost", 
    port="5432"
)

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

@app.route('/eye_test')
def eye_test():
    return render_template('eye_test.html')

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

@app.route('/eye_distance')
def eye_distance():
    return render_template('eye_distance.html')

# 初始化全局變量
time_remaining = 5
depth_value = 0

#視力檢測 測距離
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global time_remaining, depth_value
        
        try:
            # 設置深度與彩色流
            pipeline = rs.pipeline()
            rs_config = rs.config()

            # 啟用攝影機的深度與彩色流
            rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # 開始串流
            pipeline.start(rs_config)
            align_to = rs.stream.color
            align = rs.align(align_to)

            # 初始化MTCNN
            color = (0,255,0)
            minsize = 20  # 偵測人臉的最小尺寸
            threshold = [0.6, 0.7, 0.7]  # 三階段門檻
            factor = 0.709  # 縮放因子
            depth_value = 0  # 記錄眼睛中心的深度距離
            start_time = None

            # 建立TensorFlow圖與會話
            with tf.Graph().as_default():
                config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
                config.gpu_options.per_process_gpu_memory_fraction = 0.5
                sess = tf.compat.v1.Session(config=config)
                with sess.as_default():
                    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

            frame_counter = 0  # 增加跳幀機制
            while True:
                frames = pipeline.wait_for_frames()
                frame_counter += 1
                if frame_counter % 5 != 0:  # 每五幀處理一次來減少負載
                    continue

            
                # 取得RealSense畫面
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # 將影像轉換為NumPy陣列
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # MTCNN人臉偵測
                bounding_boxes, points = detect_face.detect_face(color_image, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]

                if nrof_faces > 0:
                    if start_time is None:
                        start_time = time.time()  # 記錄第一次偵測到人臉的時間

                    # 取得眼睛位置
                    points = np.array(points).transpose([1, 0]).astype(np.int16)

                    det = bounding_boxes[:, 0:4]#(左上角 x, 左上角 y, 右下角 x, 右下角 y)
                    det_arr = []
                    img_size = np.asarray(color_image.shape)[0:2]#得到圖像的高度和寬度
                    detect_multiple_faces=False#處理多個檢測到的人臉或僅處理其中的一個

                    det_arr.append(np.squeeze(det))#det_arr 中的每個元素都是一個表示邊界框的一維陣列

                    det_arr = np.array(det_arr)
                    det_arr = det_arr.astype(np.int16)

                    for i, det in enumerate(det_arr):#遍歷 det_arr 中的每個邊界框
                        if len(det) > 0  and len(det) == 4:
                            cv2.rectangle(color_image, (det[0],det[1]), (det[2],det[3]), color, 2)#在原始影像上繪製一個矩形

                        #在人臉上繪製 5 個特徵點
                        facial_points = points[i]
                        for j in range(0,5,1):
                            #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
                            cv2.circle(color_image, (facial_points[j], facial_points[j + 5]), 2, (0, 0, 255), -1, 1)

                        # 取出左右眼的位置座標
                        left_eye_x, left_eye_y = facial_points[0], facial_points[1]
                        right_eye_x, right_eye_y = facial_points[2], facial_points[3]
                        # print("Left eye coordinates:", (left_eye_x, left_eye_y))
                        # print("Right eye coordinates:", (right_eye_x, right_eye_y))
                        # print("eye_center_x",(right_eye_x + left_eye_x)//2)
                        # print("eye_center_y",(right_eye_y + left_eye_y)//2)
                        eye_center_x=(right_eye_x + left_eye_x)//2
                        eye_center_y=(right_eye_y + left_eye_y)//2

                        if 0 <= eye_center_x < depth_image.shape[1] and 0 <= eye_center_y < depth_image.shape[0]:
                            # 從深度影像中獲取眼睛中心點的深度值
                            depth_value = depth_frame.get_distance(eye_center_y, eye_center_x)
                            # 眼睛中心點與相機的距離
                            print("Distance from camera to eye center (in meters):", depth_value)
                        else:
                            print("Eye center point is out of bounds of depth image.")

                    # 計算已經偵測到人臉的時間
                    elapsed_time = time.time() - start_time
                    time_remaining = max(0, 5 - int(elapsed_time))  # 更新剩餘時間

                    if elapsed_time >= 5:
                        break  # 如果超過5秒，停止串流

                else:
                    start_time = None  # 沒有偵測到人臉時，重置計時

                # 將畫面編碼為JPEG格式，傳送到前端
                ret, buffer = cv2.imencode('.jpg', color_image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # 結束串流
            pipeline.stop()

        except Exception as e:
            print(f"Error: {e}")

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 視力檢測 傳送倒數時間和最終量測的深度值
@app.route('/data_feed')
def data_feed():
    
    return jsonify(time=time_remaining, depth=round(depth_value, 2))


@app.route('/start_eye_dis', methods=['POST'])
def start_eye_dis():
    try:
        # Configure depth and color streams
        pipeline = rs.pipeline()
        rs_config = rs.config()
        
        # Initialize MTCNN
        minsize = 20  # Minimum size of the face
        threshold = [0.6, 0.7, 0.7]  # Three-step threshold
        factor = 0.709  # Scale factor
        color = (0, 255, 0)
        

        
        with tf.Graph().as_default():
            config = tf.compat.v1.ConfigProto(log_device_placement=True, allow_soft_placement=True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            sess = tf.compat.v1.Session(config=config)
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        # Setup RealSense Camera
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = rs_config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            return jsonify({"status": "error", "message": "This demo requires a camera with a color sensor."})

        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(rs_config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        start_time = None  # To track when the face is first detected
        depth_value = 0    # To store the distance to the face

        # Main loop
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Face detection
            bounding_boxes, points = detect_face.detect_face(color_image, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces > 0:
                if start_time is None:
                    start_time = time.time()  # Start the timer when the face is first detected

                # Get facial points and calculate eye center
                points = np.array(points)
                points = np.transpose(points, [1, 0])
                points = points.astype(np.int16)
                
                left_eye_x, left_eye_y = points[0][0], points[0][1]
                right_eye_x, right_eye_y = points[0][2], points[0][3]
                eye_center_x = (left_eye_x + right_eye_x) // 2
                eye_center_y = (left_eye_y + right_eye_y) // 2

                if 0 <= eye_center_x < depth_image.shape[1] and 0 <= eye_center_y < depth_image.shape[0]:
                    depth_value = depth_frame.get_distance(eye_center_y, eye_center_x)  # Get depth value of eye center

                # Check if 15 seconds have passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= 15:
                    break  # Exit the loop after 15 seconds

            else:
                start_time = None  # Reset the timer if no face is detected

        # Stop streaming
        pipeline.stop()

        # Return the final distance value
        return jsonify({
            "status": "success",
            "message": f"Face detected. Distance to eye center: {depth_value} meters"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

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

#電腦端色盲點圖功能顯示結果
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

    answer = user_ans.query.filter_by(user_id=user_id, id=index).first()
    id_for__question = answer.question_id #從使用者作答圖片中獲取題目的編號
    if not answer or not answer.image_data:
        return jsonify({'error': 'Answer image not found'}), 404

    # 從資料庫獲取題目圖片
    question = pic.query.filter_by(id=id_for__question).first()
    if not question or not question.image_data:
        print("NOOOOOOOOOOOOOOOOOOO\n")
        # return jsonify({'error': 'Question image not found'}), 404
    else:
        print("Yessssssssssssssssssssss\n")
    
    #將使用者作答圖片和題目圖片疊在一起
    #使用 PIL 打開作答圖片和題目圖片
    answer_image = Image.open(BytesIO(answer.image_data))
    question_image = Image.open(BytesIO(question.image_data))

    #確保兩張圖片尺寸相同，如果不同則調整大小
    if question_image.size != answer_image.size:
        answer_image = answer_image.resize(question_image.size)

    # 如果answer_image 没有透明度通道，先轉換為'RGBA'
    if answer_image.mode != 'RGBA':
        answer_image = answer_image.convert('RGBA')

    # 如果question_image不是 'RGBA'，也轉換為'RGBA'
    if question_image.mode != 'RGBA':
        question_image = question_image.convert('RGBA')

    # 將answer_image 交叠在 question_image 上，黏貼時使用answer_image的透明度作為mask
    question_image.paste(answer_image, (0, 0), answer_image)

    # 显示合成后的图片
    # question_image.show()

    if question_image:

        # 確保 question_image是PIL.Image對象
        if question_image:
            # 如果图像是 'RGBA' 模式，先转换为 'RGB'，去除透明通道
            if question_image.mode == 'RGBA':
                question_image = question_image.convert('RGB')

        # 创建一个字节流缓冲区
        buffered = BytesIO()

        # 將圖像保存到字節流緩衝區中，格式為 JPEG（或根據你的圖像格式調整）
        question_image.save(buffered, format="JPEG")

        #獲取字節流中的二進制數據
        image_bytes = buffered.getvalue()

        #將二進制數據轉換為 base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        #生成 Base64 URL，指定圖像格式為JPEG
        image_url = f"data:image/jpeg;base64,{encoded_image}"

        return jsonify({'image_url': image_url})  # 直接返回單一圖片的 URL
        
    else:
        return jsonify({'error': 'Image not found'}), 404

# 計算色盲點圖分數
@app.route('/calculate-score', methods=['POST'])
def calculate_score():
    print("進來了 !!!!!!!!!!!!!!!")
    width = 12  #可容許誤差寬度
    data = request.get_json()
    user_id = data.get('user_id')
    final_score = 0.0

    if not user_id:
        return jsonify({'error': 'User ID not provided'}), 400

    try:
        for i in range(1, 11):  # 假設有 10 個問題
            id = i
            # 從資料庫獲取答案和用戶提交的圖像
            # get_image_from_db
            user_answer = db.session.query(user_ans).filter_by(user_id=user_id, id=id).first()
            question_id = user_answer.question_id
            answer_image = db.session.query(ans).filter_by(id_ans_cb=question_id).first()

            if not user_answer:
                # 如果用戶的回答不存在，記錄錯誤並繼續
                print(f"User answer not found for question {question_id}")
                continue

            if not answer_image:
                # 如果答案不存在，记录错误并继续
                print(f"Answer image not found for question {question_id}")
                continue
            
            # 從資料庫中提取二進制數據
            image_user_data = user_answer.image_data
            image_ans_data = answer_image.image_data

            # 將二進制數據轉換為 NumPy 數組
            image_user_array = np.frombuffer(image_user_data, np.uint8)
            image_ans_array = np.frombuffer(image_ans_data, np.uint8)

            # 使用 OpenCV 解碼為圖片
            image_user = cv2.imdecode(image_user_array, cv2.IMREAD_UNCHANGED)
            image_ans = cv2.imdecode(image_ans_array, cv2.IMREAD_UNCHANGED)

            # 確認圖片是否解碼成功
            if image_user is None or image_ans is None:
                print(f"Failed to decode images for question {question_id}")
                continue

            # 調整圖像大小，使它們相同
            height, width = image_ans.shape[:2]  # 獲取答案圖像的尺寸
            # print(height, width)

            image_user_resized = cv2.resize(image_user, (width, height))  # 调整用户圖像到相同大小
            
            # 計算每一張圖像的分數
            score = Score_calculation(image_ans, image_user_resized)
            # score = Score_calculation(image_ans, image_user)
            # print(score)
            final_score += score
            # print(final_score)

        # 確認是否至少成功處理了一些問題
        if final_score == 0.0:
            return jsonify({'error': 'No valid answers or images found'}), 400
        
        average_final_score = final_score / 10.0  # 計算平均分數
        print(f"User {user_id} - Average Score: {average_final_score}")
        return jsonify({'score': average_final_score})
    
    except Exception as e:
        print(f"Error calculating score for user {user_id}: {e}")
        return jsonify({'error': 'Failed to calculate score'}), 500


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
    # 獲取 completedQuestions 數據
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
    
# finish.html 禁止返回上一頁的功能
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
    user_uuid = request.args.get('session')  # 從查詢参數中獲取session ID
    if user_uuid:
        return render_template('handwrite.html', user_uuid=user_uuid)
    else:
        return "User UUID not provided", 400

@app.route('/color_blind_spot_map')
def color_blind_spot_map():
    user_uuid = request.args.get('session')  # 從查詢参數中獲取session ID
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

#生成醫囑
@app.route('/generate-advice', methods=['POST'])
def generate_advice():
    data = request.json
    symptoms = data.get('symptoms')
    print(symptoms)
    # 使用 Ollama CLI 調用 Llama 3 來生成醫囑
    try:
        prompt = f"請根據以下症狀生成，一段約300字的中文醫療建議，不需要講太多細節，要中文的{symptoms}"
        result = subprocess.run(
            ['ollama', 'run', 'llama3', ], input=prompt,
            capture_output=True, text=True, #stderr=subprocess.PIPE,
            encoding='utf-8',  # 指定使用 utf-8 編碼
            errors='ignore'    # 忽略無法編碼的字符
        )
        if result.stderr:
            app.logger.error(f"Subprocess error: {result.stderr}")
        advice = result.stdout.strip()
        return jsonify({'advice': advice})

    except Exception as e:
            app.logger.error(f"Exception: {e}")
            return jsonify({'error': str(e)}), 500

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
