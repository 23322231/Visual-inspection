from flask import Flask, render_template,request, jsonify
from flask_socketio import SocketIO, send
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
# socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quiz')
def start():
    return render_template('quiz1.html')

@app.route('/camera')
def choose():
    return render_template('camera_try.html')

@app.route('/myopia')
def confirm():
    return render_template('myopia.html')

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


# @app.route('/upload', methods=['POST'])
# def upload_image():
#     data = request.json  # 從 POST 請求中獲取 JSON 格式的數據(前端傳來的圖片資料)
#     # 檢查是否有圖片資料被傳遞過來
#     if 'image' not in data:
#         return jsonify({'error': 'No image data found'})
    
#     image_data = data['image']  # 從 JSON 數據中提取名為 'image' 的 key 所對應的 value(前端傳來的圖片資料Base64格式字串)
#     image_data = image_data.replace('data:image/png;base64,', '')  # 去除了 DataURL 的前綴部分
#     # 將經過處理的 Base64 字符串解碼成二進制數據
#     # 然後使用 BytesIO 將其包裝成 BytesIO 對象
#     # 最後使用 Pillow 的 Image.open() 方法打開圖片，生成一個 Image
#     image = Image.open(BytesIO(base64.b64decode(image_data)))
#     image.save('uploaded_image.png')  # 將打開的圖片對象保存為 PNG 格式的圖片檔案

#     return jsonify({'message': 'Image uploaded successfully'})

# @socketio.on('button_press')
# def handle_button_press(msg):
#     print('Button Pressed:', msg)
#     socketio.emit('button_press', msg)
    

if __name__ == '__main__':
    # socketio.run(app)
    # socketio.run(app, debug=True,host='0.0.0.0', port=5000)
    app.run(debug=True)
