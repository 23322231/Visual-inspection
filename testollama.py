from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/generate-advice', methods=['POST'])
def generate_advice():
    data = request.json
    symptoms = data.get('symptoms')

    # 使用 Ollama CLI 調用 Llama 3 來生成醫囑
    try:
        result = subprocess.run(
            ['ollama', 'generate', 'llama3', '--prompt', f"根據以下症狀生成醫囑：{symptoms}"],
            capture_output=True, text=True
        )

        # 假設生成的結果是 JSON 格式，你可以解析它
        advice = result.stdout.strip()

        return jsonify({'advice': advice})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
