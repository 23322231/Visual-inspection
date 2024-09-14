from flask_sqlalchemy import SQLAlchemy
from flask import Flask, jsonify
from sqlalchemy import text

app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://april0909:c7CslksYkeusqcvAkMecoFDQPIFuiPKp@dpg-cqcj0sg8fa8c73crb3u0-a.oregon-postgres.render.com/data_uire"

db = SQLAlchemy(app)

@app.route('/')
def index():
    try:
        # 嘗試進行簡單的查詢
        sql_cmd = text("SELECT 1")
        query_data = db.session.execute(sql_cmd)
        result = [row[0] for row in query_data]

        if result:
            return jsonify({"status": "success", "message": "Database connection successful", "result": result})
        else:
            return jsonify({"status": "error", "message": "No data returned from test query"})

    except Exception as e:
        # 捕獲並返回錯誤
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
