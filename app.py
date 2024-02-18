from flask import Flask, render_template
from flask_socketio import SocketIO, send

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/confirm')
def confirm():
    return render_template('confirm.html')

@app.route('/images')
def image():
    return render_template('images2.html')

@socketio.on('button_press')
def handle_button_press(msg):
    print('Button Pressed:', msg)
    socketio.emit('button_press', msg)
    

if __name__ == '__main__':
    socketio.run(app, debug=True)
