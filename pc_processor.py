import cv2
import socket
import struct
import numpy as np
from ultralytics import YOLO
import time
import json
import os
import easyocr
import requests
import base64
import threading


# 初始化 EasyOCR 阅读器（只加载一次，支持简体中文和英文）
print("正在加载 EasyOCR 车牌识别模型...")
plate_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)  # gpu=False 强制使用 CPU
print("EasyOCR 加载完成")
def get_access_token():
  
    params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY
    }
    return requests.post(url, params=params).json().get("access_token")

def recognize_plate(vehicle_img):
    try:
        # 将图像转为 base64
        _, encoded = cv2.imencode('.jpg', vehicle_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(encoded).decode('utf-8')
        
      
        payload = {"image": img_base64}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        response = requests.post(url, data=payload, headers=headers)
        result = response.json()
        
        if "words_result" in result:
            plate = result["words_result"]["number"]
            print(f"✅ 百度API识别成功: {plate}")
            return plate
        else:
            print(f"⚠️ 识别失败: {result.get('error_msg', '未知错误')}")
            return None
    except Exception as e:
        print(f"❌ API调用异常: {e}")
        return None

# 初始化
# ========== 配置 ==========
SERVER_IP = '172.20.10.4'        # 树莓派实际IP，请根据情况修改
SERVER_PORT = 8000
MODEL_PATH = 'yolov8n.pt'
CONF_THRESH = 0.25
CLASSES = [2, 5, 7]              # 2:car, 5:bus, 7:truck

# 输出目录（桌面上的 yolo_output 文件夹）
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "yolo_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 最新检测画面路径
LATEST_IMAGE_PATH = os.path.join(OUTPUT_DIR, "latest.jpg")

# ========== 全局变量 ==========
no_parking_zones = []            # 存储禁停区（矩形或多边形）
# 高德地图 API 配置（替换成你的 Key）
AMAP_KEY = '08b2614e997dbe9b02a62574748c328a'   # 你截图中的 Key
AMAP_URL = 'https://restapi.amap.com/v3/geocode/regeo'

latest_gps = (None, None)
latest_address = None          # 新增：存储最新地址
gps_lock = threading.Lock()
last_save_time = {}              # 记录上次保存时间，避免重复
enter_time = {}                  # 记录车辆首次进入禁停区的时间

# ========== 辅助函数 ==========
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def is_violation(bbox, zones):
    """判断车辆中心点是否在任意禁停区内"""
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    for zone in zones:
        if point_in_polygon((cx, cy), zone):
            return True
    return False

    
def gps_receiver():
    global latest_gps
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 8002))
    server.listen(1)
    print("等待树莓派GPS数据连接...")
    conn, addr = server.accept()
    print(f"GPS已连接: {addr}")
    buffer = ""
    while True:
        try:
            data = conn.recv(1024).decode()
            if not data:
                break
            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    gps = json.loads(line)
                    lat, lon = gps['latitude'], gps['longitude']
                  
                    address = reverse_geocode(lat, lon)
                    with gps_lock:
                        latest_gps = (lat, lon)
                        latest_address = address
                        print(f"GPS 更新: ({lat}, {lon}) -> 地址: {addr}")
                      
        except Exception as e:
            print(f"GPS接收错误: {e}")
            break

def generate_report(vehicle_class, confidence, bbox, timestamp, plate_number=None, latitude=None, longitude=None, address=None):
    # 强制转换所有可能为 tensor 或 set 的类型
    vehicle_class = str(vehicle_class)
    confidence = float(confidence)
    # bbox 可能是 list 或 tensor，确保转为 list of floats
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()
    bbox = [float(x) for x in bbox]
    plate_number = str(plate_number) if plate_number else "未识别"
    
    report = {
        "timestamp": timestamp,
        "vehicle_type": vehicle_class,
        "confidence": confidence,
        "plate_number": plate_number,
        "position": {
            "x": int((bbox[0] + bbox[2]) / 2),
            "y": int((bbox[1] + bbox[3]) / 2)
        },
        "bbox": bbox,
        "description": f"检测到{vehicle_class}（车牌：{plate_number}）违停在禁停区域。",
        "action": "记录并通知车主"
    }
    if latitude is not None and longitude is not None:
        report["location"] = {"latitude": latitude, "longitude": longitude}
    if address:
        report["address"] = address 
    return report

def save_violation(frame, report):
    timestamp = report['timestamp']
    img_path = os.path.join(OUTPUT_DIR, f"violation_{timestamp}.jpg")
    json_path = os.path.join(OUTPUT_DIR, f"violation_{timestamp}.json")
    cv2.imwrite(img_path, frame)
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"违停记录已保存: {img_path}  车牌: {report['plate_number']}")
    except Exception as e:
        print(f"保存 JSON 失败: {e}")
# ========== 鼠标回调（绘制矩形禁停区）==========
rect_start = None
rect_end = None
drawing_rect = False

def mouse_callback(event, x, y, flags, param):
    global rect_start, rect_end, drawing_rect, no_parking_zones
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        drawing_rect = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing_rect:
        rect_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_rect = False
        rect_end = (x, y)
        if rect_start and rect_end:
            x1, y1 = rect_start
            x2, y2 = rect_end
            # 将矩形转换为多边形（四个点）
            zone = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            no_parking_zones.append(zone)
            print(f"已添加矩形禁停区: ({x1},{y1}) -> ({x2},{y2})")
        rect_start = None
        rect_end = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        no_parking_zones.clear()
        print("已清除所有禁停区")

# ========== 主函数 ==========
def main():
    global rect_start, rect_end, drawing_rect, last_save_time, enter_time

    print("加载YOLO模型...")
    model = YOLO(MODEL_PATH)
    print("模型加载完成")

    print(f"连接树莓派 {SERVER_IP}:{SERVER_PORT} ...")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER_IP, SERVER_PORT))
    # 启动GPS接收线程
    #gps_thread = threading.Thread(target=gps_receiver, daemon=True)
    #gps_thread.start()
    #print("已连接到树莓派")

    window_name = "YOLO车辆检测 + 违停监控 (按住左键拖拽画矩形禁停区，右键清除)"
    # 添加预设禁停区（下半部分）
    preset_zone = [(0, 240), (640, 240), (640, 480), (0, 480)]
    no_parking_zones.append(preset_zone)
    print("已添加预设禁停区: 下半部分矩形")
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    data = b""
    payload_size = 4
    frame_count = 0
    start_time = time.time()

    print("\n操作说明：")
    print("- 按住鼠标左键拖拽：绘制矩形禁停区")
    print("- 鼠标右键：清除所有禁停区")
    print("- 按 'q' 退出，按 's' 手动截图\n")

    while True:
        # 接收图像长度
        while len(data) < payload_size:
            packet = client.recv(4096)
            if not packet:
                break
            data += packet
        if len(data) < payload_size:
            break
        img_size = struct.unpack('I', data[:4])[0]
        data = data[4:]

        # 接收图像数据
        while len(data) < img_size:
            data += client.recv(4096)
        img_data = data[:img_size]
        data = data[img_size:]

        # 解码
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # YOLO检测（仅车辆）
        results = model(frame, conf=CONF_THRESH, classes=CLASSES)
        annotated = results[0].plot()

        # 绘制已添加的禁停区（红色半透明）
        overlay = annotated.copy()
        for zone in no_parking_zones:
            cv2.fillPoly(overlay, [np.array(zone, np.int32)], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
        for zone in no_parking_zones:
            cv2.polylines(annotated, [np.array(zone, np.int32)], True, (0, 0, 255), 2)

        # 绘制正在拖拽的矩形预览
        if drawing_rect and rect_start and rect_end:
            cv2.rectangle(annotated, rect_start, rect_end, (0, 255, 255), 2)

        # 显示提示文字
        cv2.putText(annotated, f"Zones: {len(no_parking_zones)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, "Left drag: draw rectangle | Right click: clear zones",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 违停检测（只针对车辆）
        for r in results[0].boxes:
            bbox = r.xyxy[0].tolist()
            cls = int(r.cls[0])
            conf = float(r.conf[0])
            class_name = model.names[cls]  # car, bus, truck
            x1, y1, x2, y2 = map(int, bbox)

            # 生成车辆唯一ID
            vehicle_id = f"{class_name}_{(x1 + x2) // 20}_{(y1 + y2) // 20}"
            now = time.time()

            if is_violation(bbox, no_parking_zones):
                # 车辆在禁停区内
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, "VIOLATION", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 记录首次进入时间
                if vehicle_id not in enter_time:
                    enter_time[vehicle_id] = now
                else:
                    # 停留超过3秒且距离上次保存超过10秒
                    if now - enter_time[vehicle_id] >= 3:
                        if vehicle_id not in last_save_time or (now - last_save_time[vehicle_id]) > 10:
                            last_save_time[vehicle_id] = now
                            # 裁剪车辆区域并识别车牌
                            vehicle_crop = frame[y1:y2, x1:x2]
                            if vehicle_crop.size != 0:
                                plate_number = recognize_plate(vehicle_crop)
                            else:
                                plate_number = None
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            with gps_lock:
                                 lat, lon = latest_gps   # 假设 latest_gps 是一个 (lat, lon) 元组
                                 address = latest_address
                            report = generate_report(class_name, conf, bbox, timestamp, plate_number, lat, lon, address)
                            save_violation(frame, report)
            else:
                # 车辆不在禁停区内
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, class_name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # 清除进入时间记录
                enter_time.pop(vehicle_id, None)

        # 保存最新检测画面（供Web服务器使用）
        cv2.imwrite(LATEST_IMAGE_PATH, annotated)

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            manual_path = os.path.join(OUTPUT_DIR, f"manual_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(manual_path, frame)
            print(f"手动截图已保存: {manual_path}")

    client.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


    
