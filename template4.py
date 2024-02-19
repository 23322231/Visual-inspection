from flask import Flask, request, jsonify
from flask import render_template
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('handwrite1.html')

# 在 /receive_coordinates 路由上監聽來自客戶端的 POST 請求
@app.route('/receive_coordinates', methods=['POST'])
def receive_coordinates():
    data = request.get_json()   # 從 POST 請求中取得 JSON 格式的資料
    print("Received coordinates:", data)
    # 返回一個 JSON 格式的回應，其中包含一個名為 message 的屬性，指示座標資料已成功接收
    return jsonify({'message': 'Coordinates received successfully'})

if __name__ == '__main__':
    app.run(debug=True)

