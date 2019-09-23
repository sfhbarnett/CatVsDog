from  torch.utils.data import Dataset
import torch
from PIL import Image
import os

class CatVsDog(Dataset):

    def __init__(self,filelist,root_dir,transform=None):

        self.images = filelist
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.images[idx])
        image = Image.open(img_name)
        annotation = self.images[idx][0:3]
        sample = {'image':image, 'annotation':annotation}
        return sample

