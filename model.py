import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as f
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)
vgg.load_state_dict(torch.load('pretrained/vgg.pth'))
class VggEncoder(nn.Module):
    def __init__(self):
        super(VggEncoder,self).__init__()
        self.style1=nn.Sequential()
        self.style2=nn.Sequential()
        self.style3 = nn.Sequential()
        self.style4 = nn.Sequential()
        for i in range(4):
            self.style1.add_module("style1_"+str(i),vgg[i])
        for i in range(4,11):
            self.style2.add_module("style2_"+str(i),vgg[i])
        for i in range(11,18):
            self.style3.add_module("style3_"+str(i),vgg[i])
        for i in range(18,31):
            self.style4.add_module("style4_"+str(i),vgg[i])
        for parameter in self.parameters():
            parameter.requires_grad=False;
    def forward(self,x):
        style1 = self.style1(x)
        style2 = self.style2(style1)
        style3 = self.style3(style2)
        style4 = self.style4(style3)
        return style1,style2,style3,style4


decoder=nn.Sequential(
    nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(256, 256, kernel_size=(3, 3),stride=(1,1),padding=(1,1)),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=(3, 3),stride=(1,1),padding=(1,1)),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=(3, 3),stride=(1,1),padding=(1,1)),
    nn.ReLU(),
    nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 3, kernel_size=(3, 3),stride=(1,1),padding=(1,1)),

)


class styleNet(nn.Module):
    def __init__(self):
        super(styleNet,self).__init__()
        self.encoder=VggEncoder()
        for parameter in self.encoder.parameters():
            parameter.requires_grad=False;
        self.decoder=decoder
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
        c_feature=c_features[3]
        s_feature=s_features[3]
        output_feature=self.adaIN(c_feature,s_feature)
        output_feature=output_feature*control+(1-control)*c_feature
        output_image=self.decoder(output_feature)
        return output_image,output_feature,s_features
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
        output_image,output_feature,s_features=self.style_transform(content,style,control)

        return output_image,output_feature,s_features
    def get_loss(self,output_image,output_feature,s_features):
        output_image_features=self.encoder(output_image)
        content_loss=self.content_loss(output_image_features[3],output_feature)
        style_loss=self.style_loss(output_image_features,s_features)
        return output_image,content_loss,style_loss



