import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import imageio
import time
from torchvision import transforms
from t_model import *
from torchvision.utils import save_image
device= "cuda" if torch.cuda.is_available() else "cpu"
styleTransformer=styleSNet().to(device)
styleTransformer.load_state_dict(torch.load('trained_s5_model_0_80000.pth'))
style_path='style/sketch.png'
style_img = Image.open(style_path)

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
content_tf = test_transform((512,512), 256)
style_tf = test_transform((512,512), 256)
input='tt.jpg'
outputfile='test.jpg'
content=content_tf(Image.open(input))
style=style_tf(style_img)
style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)
control=0.8
with torch.no_grad():
    outputs = styleTransformer(content, style, control)
    output = outputs[0]
output = output.cpu()
output = output.squeeze(0)
output =transforms.ToPILImage()(output)
output.save(outputfile)

