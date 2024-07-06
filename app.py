# 統整版 用這個啟動
from flask import Flask, render_template,request, jsonify
from flask_socketio import SocketIO, send,emit
import base64
from PIL import Image
from io import BytesIO
# 引入 Flask 和 PyMongo 套件
# from flask_pymongo import PyMongo
# from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
socketio = SocketIO(app)
# 設定 MongoDB 連線資訊
# app.config["MONGO_URI"] = "mongodb://localhost:27017/dorm"
# mongo = PyMongo(app)
# socketio = SocketIO(app)

# 建立 HTTPBasicAuth 物件
# auth = HTTPBasicAuth()

# 設定帳號密碼驗證機制
# @auth.verify_password
# def verify_password(username, password):
#     return username == "admin" and password == "123456"

# 定義路由和 HTTP 方法，並使用 auth.login_required decorator要求驗證
# @app.route("/visitors", methods=["GET", "POST"])
# @auth.login_required
# def visitors():

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
    return render_template('handwrite.html')

@app.route('/elements')
def elements():
    return render_template('elements.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/qrcodehandwrite')
def handwrite():
    return render_template('handwrite.html')

@app.route('/show')
def show():
    return render_template('show.html')

# @app.route('/show_image')
# def image():
#     return render_template('show_image.html')

# # 在 /receive_coordinates 路由上監聽來自客戶端的 POST 請求
# @app.route('/receive_coordinates', methods=['POST'])
# def receive_coordinates():
#     data = request.get_json()   # 從 POST 請求中取得 JSON 格式的資料
#     print("Received coordinates:", data)
#     # 返回一個 JSON 格式的回應，其中包含一個名為 message 的屬性，指示座標資料已成功接收
#     return jsonify({'message': 'Coordinates received successfully'})


# 在 /receive_coordinates 路由上監聽來自客戶端的 POST 請求
@app.route('/receive_coordinates', methods=['POST'])
def receive_coordinates():
    data = request.get_json()   # 從 POST 請求中取得 JSON 格式的資料
    print("Received coordinates:", data)
    # 返回一個 JSON 格式的回應，其中包含一個名為 message 的屬性，指示座標資料已成功接收
    return jsonify({'message': 'Coordinates received successfully'})

# @socketio.on('button_press')
# def handle_button_press(msg):
#     print('Button Pressed:', msg)
#     socketio.emit('button_press', msg)
# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# @socketio.on('start_drawing')
# def handle_start_drawing():
#     emit('start_drawing', broadcast=True)

# @socketio.on('submit_drawing')
# def handle_submit_drawing(image_data):
#     emit('display_drawing', image_data, broadcast=True)

if __name__ == '__main__':
    # socketio.run(app)
    # socketio.run(app, debug=True,host='0.0.0.0', port=5000)
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)  #host改為0.0.0.0,讓手機暫時可以連上
    # app.run(host='0.0.0.0') 
