
from flask import Flask, request, jsonify
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('handwrite1.html')


@app.route('/receive_coordinates', methods=['POST'])
def receive_coordinates():
    data = request.get_json()
    print("Received coordinates:", data)
    return jsonify({'message': 'Coordinates received successfully'})

if __name__ == '__main__':
    app.run(debug=True)
