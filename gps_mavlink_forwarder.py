#!/usr/bin/env python3
import time
import socket
import json
import serial
from pymavlink import mavutil

# ========== 配置参数 ==========
# USB 直连飞控的设备文件（通常是 /dev/ttyACM0 或 /dev/ttyUSB0）
FC_SERIAL_PORT = '/dev/ttyACM0'
FC_BAUD = 115200          # 与飞控 USB 端口的波特率一致（常用 115200）

# 电脑服务器的 IP 和端口（运行 gps_server.py 的电脑）
PC_SERVER_IP = '172.20.10.3'   # 替换为你的电脑实际局域网 IP
PC_SERVER_PORT = 8001

# ========== 建立 MAVLink 连接 ==========
print(f"Connecting to flight controller on {FC_SERIAL_PORT} at {FC_BAUD} baud...")

# 注意：pymavlink 的 mavutil.mavlink_connection 可以直接使用串口路径
master = mavutil.mavlink_connection(FC_SERIAL_PORT, baud=FC_BAUD)

# 等待心跳包，确认连接成功
print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Connected to flight controller (system {master.target_system}, component {master.target_component})")

# 可选：请求飞控以 1Hz 频率发送 GPS 原始数据（GLOBAL_POSITION_INT）
# 这条命令不是必须的，因为飞控通常会主动广播 GPS 数据。如果收不到数据可以取消注释。
# master.mav.command_long_send(
#     master.target_system, master.target_component,
#     mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
#     0, 33, 1000000, 0, 0, 0, 0, 0
# )

def send_gps_to_server(lat, lon, alt, timestamp):
    """通过 TCP 将 GPS 数据发送给电脑服务器"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((PC_SERVER_IP, PC_SERVER_PORT))
        gps_data = {
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "timestamp": timestamp
        }
        sock.sendall((json.dumps(gps_data) + "\n").encode())
        sock.close()
        print(f"Sent GPS: {gps_data}")
    except Exception as e:
        print(f"Failed to send GPS: {e}")

print("Listening for GPS data...")
while True:
    # 接收 MAVLink 消息（类型为 GLOBAL_POSITION_INT）
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    if msg:
        # 转换坐标（MAVLink 使用 1e7 倍整数）
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0   # 相对高度（米）
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"GPS: lat={lat:.7f}, lon={lon:.7f}, alt={alt:.2f}m")
        send_gps_to_server(lat, lon, alt, timestamp)
    else:
        # 如果没有收到指定消息，短暂休眠避免 CPU 100%
        time.sleep(0.01)




        