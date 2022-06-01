import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import time
import imageio
from torchvision import transforms
from model import *
from torchvision.utils import save_image
device= "cuda" if torch.cuda.is_available() else "cpu"
styleTransformer=styleNet().to(device)
styleTransformer.load_state_dict(torch.load('trained_model_0_15000.pth'))
def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

content_video = cv2.VideoCapture('camera.mp4')
fps = int(content_video.get(cv2.CAP_PROP_FPS))
content_video_length = int(content_video.get(cv2.CAP_PROP_FRAME_COUNT))
output_width = int(content_video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(content_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path='output.mp4'
writer = imageio.get_writer(output_video_path, mode='I', fps=fps)
style_path='style/starry_night.jpg'
style_img = Image.open(style_path)
content_tf = test_transform((512,512), 256)
style_tf = test_transform((512,512), 256)
control=0.8
start=time.time()
while(True):
    ret, content_img = content_video.read()
   # print(content_img)

    if not ret:
        break
    content = content_tf(Image.fromarray(content_img))
    style = style_tf(style_img)
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
        outputs=styleTransformer(content,style,control)
        output=outputs[0]
    output = output.cpu()
    output = output.squeeze(0)
    
    output = np.array(output*255)

    
    output = np.transpose(output, (1,2,0))
    output = cv2.resize(output, (output_width, output_height), interpolation=cv2.INTER_CUBIC)

    writer.append_data(np.array(output))
content_video.release()
end=time.time()
print(end-start)