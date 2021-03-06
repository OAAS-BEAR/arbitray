import torch.utils.data
import torch.nn.functional as f
from t_model import *
from data import *
from iter import *

learning_rate = 1e-4
epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
transformNet = VggEncoder.to(device)
transformSNet=VggSEncoder.to(device)
optimizer = torch.optim.Adam(transformSNet.parameters(), lr=learning_rate)
alpha = 1
beta = 8
omegon = 1e-6
BS = 4
control = 1.0


def main():
    # normalization=Normalization(cnn_normalization_mean,cnn_normalization_std).to(device)
    image_data = Images('/home/featurize/data/train2014')
    style_data = Images('style')
    #style_sampler = NewSampler(style_data)
    data_sampler = NewSampler(image_data)
    #style_loader = torch.utils.data.DataLoader(style_data, batch_size=BS, sampler=style_sampler)
    data_loader = torch.utils.data.DataLoader(image_data, batch_size=BS, sampler=data_sampler)
    #style_iter = iter(style_loader)
    data_iter = iter(data_loader)
    for epoch in range(epochs):
        idx = 0
        while (idx < 40000):
            image_tensor = next(data_iter)
            optimizer.zero_grad()
            # print(image_tensor.size())
            out_feature= transformNet(image_tensor)
            out_feature_s = transformSNet.forward_c(image_tensor)
            content_loss=nn.MSELoss()(out_feature[-1],out_feature_s[-1])
            style_loss=0
            for i in range(len(out_feature)):
                out_feature = out_feature[i]
                target_feature = out_feature_s[i]
                c_size = out_feature.size()
                s_size = target_feature.size()
                out_std = (out_feature.view(c_size[0], c_size[1], -1).var(dim=2) + 1e-6).sqrt().view(c_size[0],
                                                                                                     c_size[1], 1, 1)
                out_mean = out_feature.view(c_size[0], c_size[1], -1).mean(dim=2).view(c_size[0], c_size[1], 1, 1)
                target_std = (target_feature.view(s_size[0], s_size[1], -1).var(dim=2) + 1e-6).sqrt().view(s_size[0],
                                                                                                           s_size[1], 1,
                                                                                                           1)
                target_mean = target_feature.view(s_size[0], s_size[1], -1).mean(dim=2).view(s_size[0], s_size[1], 1, 1)
                style_loss += f.mse_loss(out_std, target_std) + f.mse_loss(out_mean, target_mean)
            loss=alpha*content_loss+beta*style_loss
            loss.backward()
            optimizer.step()
            if (idx + 1) % 500 == 0:
                print('epoch: %d iteration: %d loss: %.5f content_loss: %.5f style_loss: %.5f' % (
                    epoch, idx,
                    loss, alpha * content_loss, beta * style_loss))
            if (idx + 1) % 5000 == 0:
                torch.save(transformSNet.state_dict(), 'trained_model_%d_%d.pth' % (epoch, idx + 1))
            idx += 1
if __name__ == '__main__':
    main()