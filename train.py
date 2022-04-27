import torch.utils.data
import torch.nn.functional as f
from model import *
from data import *
from iter import *
learning_rate = 1e-4
epochs = 1
transformNet = styleNet().to(device);
optimizer = torch.optim.Adam(transformNet.parameters(), lr=learning_rate)
alpha = 1
beta = 5
omegon = 1e-6
BS=4
control=1.0
device= "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
def main():
    #normalization=Normalization(cnn_normalization_mean,cnn_normalization_std).to(device)
    image_data = Images('/home/featurize/data/train2014')
    style_data = Images('style')
    style_sampler=NewSampler(style_data)
    data_sampler=NewSampler(image_data)
    style_loader = torch.utils.data.DataLoader(style_data, batch_size=BS,sampler=style_sampler)
    data_loader = torch.utils.data.DataLoader(image_data, batch_size=BS,sampler=data_sampler)
    style_iter=iter(style_loader)
    data_iter=iter(data_loader)
    for epoch in range(epochs):
        idx=0
        while(idx<40000):
            
            image_tensor=next(data_iter)
            style_tensor=next(style_iter)
            optimizer.zero_grad()
            #print(image_tensor.size())
            assert(image_tensor.size()==style_tensor.size())
            output_image, out_feature, s_feature=transformNet(image_tensor,style_tensor,control)
            output_image, content_loss, style_loss=transformNet.get_loss(output_image, out_feature, s_feature)
            loss=alpha*content_loss+beta*style_loss
            loss.backward()
            optimizer.step()
            if (idx + 1) % 500 == 0:
                print('epoch: %d iteration: %d loss: %.5f content_loss: %.5f style_loss: %.5f' % (
                    epoch, idx,
                    loss, alpha * content_loss, beta * style_loss))
            if (idx + 1) % 5000 == 0:
                torch.save(transformNet.state_dict(), 'trained_model_%d_%d.pth' % (epoch, idx + 1))
            idx+=1

if __name__ == '__main__':
    main()