import socket
import time
from model import *
import torch
import numpy as np
import cv2
device= "cuda" if torch.cuda.is_available() else "cpu"
def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
import torchvision
from PIL import Image
from torchvision import transforms
styleTransformer=styleNet()
styleTransformer.load_state_dict(torch.load('trained_model_0_40000.pth'))
# create a socket object
serversocket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)

# get local machine name
host ='172.31.248.142'
print(host)

port =3389

buf_size=3000000

# bind to the port
serversocket.bind((host, port))



control=0.7
content_tf = test_transform((512, 512), 256)
style_tf = test_transform((512, 512), 256)
while True:
    serversocket.listen(5)
    clientsocket, addr = serversocket.accept()
    print("Got a connection from %s" % str(addr))
    image = bytes()

    while(len(image)!=196608):
        rec_image=clientsocket.recv(buf_size)
        image += rec_image

    style_img = np.frombuffer(image, dtype=np.uint8).reshape([256, 256, 3])
    image=bytes()
    while True:
        rec_image = clientsocket.recv(buf_size)
        image += rec_image


        if rec_image:
            rec_image = clientsocket.recv(buf_size)
            image+=rec_image
            if(len(image)==196608):
                image = np.frombuffer(image, dtype=np.uint8).reshape([256, 256, 3])
                content = content_tf(Image.fromarray(image))
                style = style_tf(style_img)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    outputs = styleTransformer(content, style, control)
                    output = outputs[0]
                output = output.cpu()
                output = output.squeeze(0)

                output = np.array(transforms.ToPILImage()(output))
                image=cv2.resize(output,(256,256))
            #print(len(image))
                clientsocket.send(image.tobytes())
                image=bytes()
        else:
            clientsocket.close()
            break