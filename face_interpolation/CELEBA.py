import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

class CELEBA(Dataset):
    def __init__(self,annotation_file, img_dir,transform=None,target_transform=None):
        self.img_labels = pd.read_table(annotation_file,sep=' ',header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(image)
        return image,label


if __name__ == '__main__':
    dataloader = DataLoader(
        CELEBA(annotation_file='./data/identity_CelebA.txt', img_dir='./data/img_align_celeba'),
        batch_size=64, shuffle=True)
    for i, (imgs, _) in enumerate(dataloader):
        print(i)
