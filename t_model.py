import os
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as f
device= "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

vgg19 = models.vgg19(pretrained=True).features.eval()
class VggLayer(nn.Module):
    def __init__(self):
        super(VggLayer,self).__init__()
        self.vgg=vgg19
        self.style1=nn.Sequential()
        self.style2=nn.Sequential()
        self.style3 = nn.Sequential()
        self.style4 = nn.Sequential()
        for i in range(4):
            self.style1.add_module("style1_"+str(i),self.vgg[i])
        for i in range(4,9):
            self.style2.add_module("style2_"+str(i),self.vgg[i])
        for i in range(9,14):
            self.style3.add_module("style3_"+str(i),self.vgg[i])
        for i in range(14,23):
            self.style4.add_module("style4_"+str(i),self.vgg[i])
        for parameter in self.parameters():
            parameter.requires_grad=False;
    def forward(self,x):
        style1 = self.style1(x)
        style2 = self.style2(style1)
        style3 = self.style3(style2)
        style4 = self.style4(style3)
        return style1,style2,style3,style4


#def content_loss(input,output):

#def style_loss(output,style):


class ConvRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super(ConvRelu,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels, kernel_size, stride,padding)
        self.insNorm=nn.InstanceNorm2d(out_channels,affine=True)
        self.activation=nn.ReLU()
    def forward(self,x):
        x=self.conv(x)
        x=self.insNorm(x)
        x=self.activation(x)
        return x

class ConvTanh(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super(ConvTanh,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding)
        self.insNorm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.Tanh()


    def forward(self, x):
        x = self.conv(x)
        x = self.insNorm(x)
        x = self.activation(x)
        return x

class Deconv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0):
        super(Deconv,self).__init__()
        self.conv=nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,padding,output_padding)
        self.insNorm=nn.InstanceNorm2d(out_channels, affine=True)
        self.activation=nn.ReLU()
    def forward(self,x):
        x=self.conv(x)
        x=self.insNorm(x)
        x=self.activation(x)
        return x

class Res(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,padding=0):
        super(Res, self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding),
            nn.InstanceNorm2d(out_channels, affine=True),
        )
        self.activation=nn.ReLU()
    def forward(self,x):
        out=self.block(x)
        return self.activation(out+x)
class styleNet(nn.Module):
    def __init__(self):
        super(styleNet,self).__init__()
        self.encoder=nn.Sequential(
            ConvRelu(3, 16, 3, 1,1),
            ConvRelu(16, 32, 3, 2,0),
            ConvRelu(32, 48, 3, 2,0),
            Res(48, 48, 3, 1,1),
            Res(48, 48, 3, 1,1),
            Res(48, 48, 3, 1,1),
            Res(48, 48, 3, 1,1),
            Res(48, 48, 3, 1,1),)
        self.decoder=nn.Sequential(
            Deconv(48, 32, 3, 2,0),
            Deconv(32, 16, 3, 2,0,1),
            ConvTanh(16, 3, 3, 1,1)
        )
        self.f_net=VggLayer()
    def adaIN(self,content,style):
        c_size=content.size()
        s_size=style.size()
        content_std=(content.view(c_size[0],c_size[1],-1).var(dim=2)+1e-6).sqrt().view(c_size[0],c_size[1],1,1).expand(c_size)
        style_std=(style.view(s_size[0],s_size[1],-1).var(dim=2)+1e-6).sqrt().view(s_size[0],s_size[1],1,1).expand(s_size)
        content_mean=content.view(c_size[0],c_size[1],-1).mean(dim=2).view(c_size[0],c_size[1],1,1).expand(c_size)
        style_mean=style.view(s_size[0],s_size[1],-1).mean(dim=2).view(s_size[0],s_size[1],1,1).expand(s_size)
        return (content-content_mean)/content_std*style_std+style_mean
    def style_transform(self,content,style,control):
        c_features=self.encoder(content)
        s_features=self.encoder(style)
        c_feature=c_features
        s_feature=s_features
        output_feature=self.adaIN(c_feature,s_feature)
        output_feature=output_feature*control+(1-control)*c_feature
        output_image=self.decoder(output_feature)
        return output_image,output_feature
    def content_loss(self,output_image,content_target):
        return f.mse_loss(output_image,content_target)
    def style_loss(self,output_image,style_target):
        loss=0
        for i in range(len(output_image)):
            out_feature=output_image[i]
            target_feature=style_target[i]
            c_size = out_feature.size()
            s_size = target_feature.size()
            out_std=(out_feature.view(c_size[0],c_size[1],-1).var(dim=2)+1e-6).sqrt().view(c_size[0],c_size[1],1,1)
            out_mean=out_feature.view(c_size[0],c_size[1],-1).mean(dim=2).view(c_size[0],c_size[1],1,1)
            target_std=(target_feature.view(s_size[0],s_size[1],-1).var(dim=2)+1e-6).sqrt().view(s_size[0],s_size[1],1,1)
            target_mean=target_feature.view(s_size[0],s_size[1],-1).mean(dim=2).view(s_size[0],s_size[1],1,1)
            loss+=f.mse_loss(out_std,target_std)+f.mse_loss(out_mean,target_mean)
        return loss
    def forward(self,content,style,control):
        output_image,output_feature=self.style_transform(content,style,control)

        return output_image,output_feature
    def get_loss(self,output_image,output_feature,style):
        output_image_features=self.f_net(output_image)
        s_features=self.f_net(style)
        content_loss=self.content_loss(output_image_features[3],output_feature)
        style_loss=self.style_loss(output_image_features,s_features)
        return output_image,content_loss,style_loss
