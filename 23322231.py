from flask import Flask, render_template, request, jsonify, session
from flask import Response,send_from_directory

app = Flask(__name__)

#點圖製作功能頁面
@app.route('/')
def elements():
    return render_template('color_blind_simulation.html')

#提供靜態檔案的路由，將內容類型指定為 JavaScript
@app.route('/static/assets/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/assets/js', filename, mimetype='text/javascript')

@app.route('/eye_echart')
def eye_echart():
    return render_template('eye_echart.html')


@app.route('/eye_distance')
def eye_distance():
    return render_template('eye_distance.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/image_test')
def image_test():
    return render_template('image_test.html')

@app.route('/image_test_result')
def image_test_result():
    return render_template('image_test_result.html')

@app.route('/finish')
def finish():
    return render_template('finish.html')

@app.route('/eye_result') #晚點調整
def eye_result():
    return render_template('eye_result.html')

@app.route('/comfirm_colordot')
def comfirm_colordot():
    return render_template('comfirm_colordot.html')



if __name__ == '__main__':
    # with app.app_context():
    app.run(host='0.0.0.0', port=5000, debug=True)