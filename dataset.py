import torch.utils.data as data
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms

class Comic_Dataset(data.Dataset):
    def __init__(self):
        self.imgs = glob(r"**/comic_faces/faces/*.jpg", recursive=True)[:256]
        self.transforms = transforms.Compose([
            transforms.Resize(256, 256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        data = self.transforms(pil_img)
        return data

    def __len__(self):
        return(len(self.imgs))

class HumanFace_Dataset(data.Dataset):
    def __init__(self):
        self.imgs = glob(r"**/LFW_faces/*.jpg", recursive=True)
        self.transforms = transforms.Compose([
            transforms.Resize(256, 256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        data = self.transforms(pil_img)
        return data

    def __len__(self):
        return(len(self.imgs))


if __name__ == "__main__":
    dataset = Comic_Dataset()
    dataset_H = HumanFace_Dataset()
    print(dataset[0].size(), dataset_H[0].size(), dataset.__len__())


    