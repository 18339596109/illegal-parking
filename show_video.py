import cv2
import socket
import struct
import numpy as np

client = socket.socket()
client.connect(('172.20.10.4', 8000))  # 改成树莓派实际IP
print("已连接")

data = b""
while True:
    # 接收4字节长度
    while len(data) < 4:
        data += client.recv(4096)
    img_size = struct.unpack('I', data[:4])[0]
    data = data[4:]
    
    # 接收图像数据
    while len(data) < img_size:
        data += client.recv(4096)
    img_data = data[:img_size]
    data = data[img_size:]
    
    # 解码并显示
    frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is not None:
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()