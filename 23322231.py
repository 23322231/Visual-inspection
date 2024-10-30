from flask import Flask, render_template, request, jsonify, session
from flask import Response,send_from_directory

app = Flask(__name__)

#點圖製作功能頁面
@app.route('/')
def elements():
    return render_template('visual_simulation.html')

#提供靜態檔案的路由，將內容類型指定為 JavaScript
@app.route('/static/assets/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/assets/js', filename, mimetype='text/javascript')


if __name__ == '__main__':
    # with app.app_context():
    app.run(host='0.0.0.0', port=5000, debug=True)