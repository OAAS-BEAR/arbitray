import torch
from torchvision import models
from model import*
vgg19 = decoder
from torchviz import make_dot
from torchsummary import summary
x=torch.rand(4,2,256,256)
summary(vgg19,input_size=(512,256,256))
