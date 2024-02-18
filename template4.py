# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:56:02 2024

@author: admin
"""

"""
2/9 chatgpt
"""
from flask import Flask, request, jsonify
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('handwrite1.html')


@app.route('/receive_coordinates', methods=['POST'])
def receive_coordinates():
    data = request.get_json()
    # 在这里处理接收到的坐标数据
    print("Received coordinates:", data)
    # 在这里执行你的后续处理逻辑，比如保存到数据库、进一步处理等
    return jsonify({'message': 'Coordinates received successfully'})

if __name__ == '__main__':
    app.run(debug=True)
