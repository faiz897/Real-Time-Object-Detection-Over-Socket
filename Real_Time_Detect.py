from torchvision.models import detection
import numpy as np
import pickle
import torch
import cv2
import socket
import base64


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCES = 0.5

CLASSES = pickle.loads(open('model/coco_classes.pickle', "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
MODEL = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=len(CLASSES),
                                           pretrained_backbone=True).to(DEVICE)
MODEL.eval()

# socket receiving
BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
host_ip = ''    # Enter Your IP
port = 9999     # Enter Your Port
socket_address = (host_ip, port)
server_socket.bind(socket_address)
print(f"listing at:{socket_address}")

while True:
    packet, _ = server_socket.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet, ' /')
    npdata = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(npdata, 1)

    cv2.imshow("intermediate", frame)

    orig = frame.copy()

    source = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    source = source.transpose((2, 0, 1))

    source = np.expand_dims(source, axis=0)
    source = source/255.0
    tensor = torch.FloatTensor(source)

    tensors = tensor.to(DEVICE)
    detections = MODEL(tensors)[0]

    for i in range(0, len(detections['boxes'])):
        confidence = detections['scores'][i]

        if confidence > CONFIDENCES:
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (start_x, start_y, end_x, end_y) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence*100:.2f}%"
            cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), COLORS[idx], 2)

            y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(orig, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("RECEIVING VIDEO", orig)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        server_socket.close()
        break
