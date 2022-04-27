from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class Images(Dataset):
    def __init__(self, filepath):
        self.dir = filepath
        self.files = os.listdir(filepath)
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename = os.path.join(self.dir, self.files[idx])
        image = Image.open(filename)
        #image = torch.from_numpy(F.resize(image, (360, 640))).to(device).permute(2, 0, 1).float()

        image=transforms.Resize(size=(512,512))(image)
        image=transforms.RandomCrop(256)(image)
        toTensor = transforms.ToTensor()
        image_tensor = toTensor(image).to(device)
        a,b,c=image_tensor.size()
        image_tensor=image_tensor.expand(3,b,c)
        return image_tensor