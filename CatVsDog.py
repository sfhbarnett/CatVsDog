from torch.utils.data import Dataset
import torch
from PIL import Image
import os

class CatVsDog(Dataset):

    def __init__(self,filelist,root_dir,transform=None,train=1):

        self.images = filelist
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.images[idx])
        image = self.transform(Image.open(img_name))
        if self.train == 1:
            annotation = self.images[idx][0:3]
            labeldict = {"cat": 0, "dog": 1}
            if annotation == 'cat' or annotation == 'dog':
                annotation = labeldict[annotation]
        else:
            annotation = self.images[idx]
        sample = {'image': image, 'annotation': annotation}
        return sample

