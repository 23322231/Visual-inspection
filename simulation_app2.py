import os
from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import base64
from color_blind_simulation import simulate_color_blindness     # 引用 color_blind_simulation.py 色覺辨識障礙功能
from gradient import calculate_edge_loss    # 引用 gradient.py 計算梯度差異並算邊緣消失比例功能

app = Flask(__name__)

# 設定上傳資料夾
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 支援的圖片格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 確保上傳資料夾存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 檢查檔案副檔名是否允許
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('color_blind_simulation.html')


@app.route('/simulation-result', methods=['GET','POST'])
def simulation_result():
    # 這些是您從 Python 計算出來的結果
    # 接收上傳的圖片文件
    file1 = request.files['original_image']
    file2 = request.files['simulated_image']
    image1 = Image.open(file1).convert('RGB')
    image2 = Image.open(file2).convert('RGB')

    # 轉為 OpenCV 格式
    img_rgb = np.array(image1)
    img_sim_rgb = np.array(image2)

    # Example usage
    edge_loss_ratio_ori, max_value = calculate_edge_loss(img_rgb, img_sim_rgb)

    return jsonify({
        'edge_loss_ratio_ori': edge_loss_ratio_ori,
        'max_value': max_value
    })


@app.route('/simulate', methods=['POST'])
def simulate():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        # 取得檔案副檔名
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        
        # 儲存檔案到伺服器的臨時目錄
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 使用 OpenCV 讀取圖片
        image = cv2.imread(filepath)

        # 根據前端選擇的色盲類型和嚴重程度進行模擬
        cb_type = int(request.form['cb_type'])
        severity = int(request.form['severity'])

        # 模擬色盲效果（這裡調用 simulate_color_blindness 函數）
        simulated_image = simulate_color_blindness(image, cb_type, severity)

        # OpenCV 的圖像處理後返回的是 NumPy 陣列，需轉換為 PIL Image 才能保存
        simulated_image_pil = Image.fromarray(simulated_image)

        # 將模擬後的圖片儲存到 BytesIO 中
        img_io = BytesIO()
        # simulated_image_pil.save(img_io, format=file_ext.upper())
        file_ext = file_ext.upper()
        if file_ext == 'JPG':
            file_ext = 'JPEG'
        simulated_image_pil.save(img_io, format=file_ext)
        img_io.seek(0)

        # 將圖片轉換為 base64 字串
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # 返回 JSON 給前端
        return jsonify({'image_data': f"data:image/{file_ext.lower()};base64,{img_base64}"})

    return 'Invalid file format', 400

if __name__ == '__main__':
    app.run(debug=True)
