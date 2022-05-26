import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import imageio
import time
from torchvision import transforms
from torchvision.utils import save_image
from t_model import *
from torchvision.utils import save_image
device = "cuda" if torch.cuda.is_available() else "cpu"
styleTransformer = styleSNet().to(device)
styleTransformer.load_state_dict(torch.load('trained_s5_model_0_80000.pth',map_location=device))
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
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
output_video_path = 'output.mp4'
style_path = 'style/sketch.png'
style_img = Image.open(style_path)
content_tf = test_transform((512, 512), 256)
style_tf = test_transform((512, 512), 256)
control = 1
i = 0
start = time.time()
while (True):
    ret, content_img = content_video.read()
    if not ret:
        break
    content = (content_tf(Image.fromarray(content_img)))
    style = (style_tf(style_img))
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
        outputs = styleTransformer(content, style, control)
        output = outputs[0]
    output = output.cpu()
    output = output.squeeze(0)
    if (i == 10):
        break
    i = i + 1
content_video.release()
end = time.time()
print(end - start)