from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import os
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import cv2
import numpy as np


class Watermark_Dataset(Dataset):
    def __init__(self, root_dir, random_crop=True, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.random_crop = random_crop
        self.transform = transform
        self.watermark_path = os.path.join(root_dir, 'data', 'frames1')
        self.org_images = glob(os.path.join(root_dir, 'data', 'masked1') + '/*.jpg')

    def __len__(self):
        return len(self.org_images)

    def __getitem__(self, index):
        label = self.org_images[index]
        img_file = os.path.join(self.watermark_path, label.split('\\')[-1].split('_')[0]+'.jpg')
        y = Image.open(img_file).convert("RGB")
        x = Image.open(label).convert("RGB")

        # Random crop
        if self.random_crop:
            # add padding if x is smaller than 256x256
            width, height = x.size
            if width < 256:
                x = cv2.copyMakeBorder(np.array(x), 0, 0, 0, 256 - width, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                y = cv2.copyMakeBorder(np.array(y), 0, 0, 0, 256 - width, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                x = Image.fromarray(x).convert("RGB")
                y = Image.fromarray(y).convert("RGB")
            if height < 256:
                x = cv2.copyMakeBorder(np.array(x), 0, 256 - height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                y = cv2.copyMakeBorder(np.array(y), 0, 256 - height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                x = Image.fromarray(x).convert("RGB")
                y = Image.fromarray(y).convert("RGB")

            i, j, h, w = transforms.RandomCrop.get_params(
                x, output_size=(256, 256))

            x = TF.crop(x, i, j, h, w)
            y = TF.crop(y, i, j, h, w)

        # to tensor
        x = transforms.ToTensor()(x)
        y = transforms.ToTensor()(y)

        # transform
        if self.transform:
            x = self.transform(x)

        return (x, y)


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

if __name__ == '__main__':
    dataset = Watermark_Dataset(os.getcwd(), random_crop=True)
    tmp = dataset[0]
    x = tmp[0]
    y = tmp[1]
    get_concat_h(x,y).show()
