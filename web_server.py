from flask import Flask, render_template, send_file, jsonify
import os
import glob
import json

app = Flask(__name__)

# 输出目录（与 pc_processor.py 保持一致）
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "yolo_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATEST_IMAGE = os.path.join(OUTPUT_DIR, "latest.jpg")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def get_image():
    """返回最新检测画面"""
    if os.path.exists(LATEST_IMAGE):
        return send_file(LATEST_IMAGE, mimetype='image/jpeg')
    else:
        return "", 404

@app.route('/violations')
def get_violations():
    pattern = os.path.join(OUTPUT_DIR, "violation_*.json")
    json_files = glob.glob(pattern)
    records = []
    for json_file in sorted(json_files, reverse=True):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)   # 这里 data 是正常定义的
        # 构造前端需要的字段
        records.append({
            "timestamp": data.get("timestamp", ""),
            "vehicle_type": data.get("vehicle_type", "未知"),
            "confidence": data.get("confidence", 0),
            "plate_number": data.get("plate_number", "未识别"),
            "location": data.get("location"),
            "image": f"violation_{data['timestamp']}.jpg" if "timestamp" in data else ""
        })
    return jsonify(records)

@app.route('/image/<path:filename>')
def serve_image(filename):
    """提供违停截图文件（因为文件在 OUTPUT_DIR 下，需要安全访问）"""
    # 仅允许访问 OUTPUT_DIR 下的 jpg 文件
    safe_path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if os.path.exists(safe_path) and safe_path.endswith('.jpg'):
        return send_file(safe_path, mimetype='image/jpeg')
    else:
        return "", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

@app.route('/save_zone', methods=['POST'])
def save_zone():
    zone_data = request.json
    # 可以写入文件或数据库，这里只打印
    print("收到规划区域:", zone_data)
    return jsonify({"status": "ok"})

@app.route('/clear_zones', methods=['POST'])
def clear_zones():
    # 清空保存的区域数据
    return jsonify({"status": "ok"})