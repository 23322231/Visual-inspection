from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
import base64
from flask import render_template

app = Flask(__name__)

# Initialize Firebase Admin SDK
# 參考:https://medium.com/@jonatanramhoj/firebase-admin-sdk-installation-guide-f64349d86a9d
# 參考:https://medium.com/bandai%E7%9A%84%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98/%E6%89%8B%E6%8A%8A%E6%89%8B%E7%A8%8B%E5%BC%8F%E5%AF%A6%E4%BD%9C%E5%88%86%E4%BA%AB%E7%B3%BB%E5%88%97-python-x-firebase-%E8%B3%87%E6%96%99%E5%BA%AB%E8%A8%AD%E5%AE%9A%E9%9B%86%E7%B0%A1%E5%96%AE%E6%93%8D%E4%BD%9C-3052a81b843a
cred = credentials.Certificate('color-blind-4ea7d-firebase-adminsdk-pitey-9e24fe342c.json')    #要改成自己的 private key(python), 新的json檔
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://color-blind-4ea7d-default-rtdb.firebaseio.com/'    #改成自己專案的APP網址
})

@app.route('/')
def index():
    return render_template('user_handwrite_image.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    image_data = data['image']

    # Decode the base64 image
    header, encoded = image_data.split(",", 1)
    image_data = base64.b64decode(encoded)

    # Save the image data to Firebase Realtime Database
    ref = db.reference('images')
    new_image_ref = ref.push({
        'image_data': image_data.decode('latin1')  # Store as latin1 to handle binary data
    })

    return jsonify({'message': 'Image uploaded successfully', 'id': new_image_ref.key}), 200

if __name__ == '__main__':
    app.run(debug=True)
