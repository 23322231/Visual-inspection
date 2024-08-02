from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import base64
from PIL import Image
from io import BytesIO
import random
import uuid
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 用於會話加密的密鑰

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://april0909:c7CslksYkeusqcvAkMecoFDQPIFuiPKp@dpg-cqcj0sg8fa8c73crb3u0-a.oregon-postgres.render.com/data_uire"

db = SQLAlchemy(app)
socketio = SocketIO(app)

# 連線資料庫的table
class pic(db.Model):
    __tablename__='images'
    id=db.Column(db.Integer,primary_key=True)
    image_data=db.Column(db.String(150))

    def __init__(self, image_data):
        self.image_data = image_data

class ans(db.Model):
    __tablename__='user_ans'
    id = db.Column(db.Integer, primary_key=True)
    image_data = db.Column(db.LargeBinary)

    def __init__(self, image_data=None):
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

@app.route('/camera')
def choose():
    return render_template('camera_try.html')

@app.route('/myopia')
def confirm():
    return render_template('myopia.html')

@app.route('/generic')
def generic():
    random_id = random.randint(1, 20)
    colorblind_test = db.session.query(pic).filter(pic.id == random_id).first()
    if colorblind_test:
        base64_data = base64.b64encode(colorblind_test.image_data).decode('utf-8')
        return render_template('handwrite.html', data=base64_data)
    else:
        return "No image found", 404

@app.route('/next-image')
def next_image():
    random_id = random.randint(1, 20)
    colorblind_test = db.session.query(pic).filter(pic.id == random_id).first()
    if colorblind_test:
        base64_data = base64.b64encode(colorblind_test.image_data).decode('utf-8')
        next_image_url = f"data:image/jpeg;base64,{base64_data}"
        return jsonify({'nextImageUrl': next_image_url})
    else:
        return jsonify({'error': 'No image found'}), 404
    
@app.route('/ishihara-test')
def elements():
    return render_template('ishihara-test.html')

@app.route('/comfirm_colordot')
def comfirm_colordot():
    return render_template('comfirm_colordot.html')

# @app.route('/handwrite')
# def handwrite():
#     user_uuid = request.args.get('user')  # 獲取查詢參數中的 user UUID
#     if user_uuid:
#         # 可以在這裡進行一些檢查或其他操作，比如驗證 UUID 格式
#         return render_template('handwrite.html', user_uuid=user_uuid)
#     else:
#         return "User UUID not provided", 400
    
@app.route('/handwrite')
def handwrite():
    user_uuid = request.args.get('session')  # 从查询参数中获取 session ID
    if user_uuid:
        # 在这里可以做进一步的处理，比如验证 session ID
        return render_template('handwrite.html', user_uuid=user_uuid)
    else:
        return "User UUID not provided", 400


@app.route('/show')
def show():
    return render_template('show.html')

@app.route('/color_blind_spot_map')
def color_blind_spot_map():
    return render_template('color_blind_spot_map.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data found'})
    
    image_data = data['image']
    image_data = image_data.replace('data:image/png;base64,', '')
    binary_image_data = base64.b64decode(image_data)

    new_image = ans(image_data=binary_image_data)
    db.session.add(new_image)
    db.session.commit()

    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image.save('uploaded_image.png')
    socketio.emit('image_uploaded', {'url': '/show'})
    return jsonify({'message': 'Image uploaded successfully'})

# @app.route('/generate-url', methods=['GET'])
# def generate_url():
#     unique_url = f"{request.host_url}handwrite?user={uuid.uuid4()}"
#     session['unique_url'] = unique_url  # 存儲到會話中
#     return jsonify({'url': unique_url})

@app.route('/generate-url', methods=['GET'])
def generate_url():
    unique_url = f"{request.host_url}handwrite?session={uuid.uuid4()}"
    session['unique_url'] = unique_url  # 存儲到會話中
    return jsonify({'url': unique_url})


@app.route('/get-url', methods=['GET'])
def get_url():
    unique_url = session.get('unique_url', None)  # 從會話中獲取 URL
    if unique_url:
        return jsonify({'url': unique_url})
    else:
        return jsonify({'error': 'URL not found'}), 404

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
