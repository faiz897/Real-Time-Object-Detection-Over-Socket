import cv2
import socket
import imutils
import base64

BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
host_ip = '192.168.56.1'
port = 9999


vid = cv2.VideoCapture(0)
fps, st, frames_to_count, cnt = (0, 0, 20, 0)

while True:
    WIDTH = 400
    while(vid.isOpened()):
        _, frame = vid.read()
        frame = imutils.resize(frame, width=WIDTH)
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        message = base64.b64encode(buffer)
        client_socket.sendto(message, (host_ip, port))
        cv2.imshow("TRANSMITTING VIDEO", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            client_socket.close()
            break