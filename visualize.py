import torch
from torchvision import models
from t_model import*
vgg19 =VggEncoder()
from torchviz import make_dot
from torchsummary import summary
x=torch.rand(4,2,256,256)
summary(vgg19,input_size=(3,256,256))
