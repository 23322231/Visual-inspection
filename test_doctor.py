from flask import Flask, request, jsonify, render_template
import subprocess
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-advice', methods=['POST'])
def generate_advice():
    data = request.json
    symptoms = data.get('symptoms')
    print(symptoms)
    # 使用 Ollama CLI調用 Llama3.2來生成醫囑
    try:
        prompt = f"你是一位只會繁體中文的臺灣醫生，請根據以下症狀生成一段約150字的全繁體中文醫療建議，不需要講太多細節，要繁體中文，臺灣用語，禁止使用英文任何單字，如果有英文單字，請翻譯官翻譯成中文{symptoms}"
        result = subprocess.run(
            ['ollama', 'run', 'llama3.2', ], input=prompt,
            capture_output=True, text=True, #stderr=subprocess.PIPE,
            encoding='utf-8',  # 指定使用 utf-8 編碼
            errors='ignore'    # 忽略無法編碼的字符
        )
        if result.stderr:
            app.logger.error(f"Subprocess error: {result.stderr}")
        advice = result.stdout.strip()
        return jsonify({'advice': advice})

    except Exception as e:
            app.logger.error(f"Exception: {e}")
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
