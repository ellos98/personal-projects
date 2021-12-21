import torch
import torch.nn as nn
import os
from glob import glob
from PIL import Image
import math
import torchvision.transforms as transforms
import numpy as np
import utils
import cyclegan_model as models
from tqdm import tqdm
import torch.nn.functional as F
import torchvision

pics = glob(os.path.join(os.getcwd(),'original')+r'\*.jpg')
pics = sorted(pics)
kernel_size = 256
STRIDE = 256 # stride
TRAIN_MEAN = [0.5109, 0.4903, 0.4257]
TRAIN_STD = [0.2509, 0.2338, 0.1879]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD)
    ]
)

model = models.cycleGan().to(DEVICE)
utils.load_checkpoint(torch.load("generator_gan.pth.tar"), model)
model.eval()

for idx, image_file in enumerate(pics):
    image = Image.open(image_file).convert('RGB')
    width, height = image.size
    max_size = math.ceil(max(width, height)/kernel_size)*kernel_size
    pad_height = max_size - height
    pad_width = max_size - width

    image = image_normalize(image).to(device=DEVICE)
    img_size = image.shape[1], image.shape[2]
    image = image.permute(1,2,0)
    kh, kw = kernel_size, kernel_size
    dh, dw = STRIDE, STRIDE # stride

    patches = image.unfold(0, kh, dh).unfold(1, kw, dw)
    patches = patches.contiguous().view(-1, 3, kh, kw)

    with torch.no_grad():
        batch_size = 64
        for id in tqdm(range(math.ceil(patches.shape[0]/batch_size)), leave=False):
            from_idx = id*batch_size
            to_idx = min((id+1)*batch_size, patches.shape[0])

            curr_patch = patches[from_idx:to_idx].to(device=DEVICE)
            patch = model(curr_patch)
            patches[from_idx:to_idx] = (patch.to("cpu"))

    patches = patches.view(1, patches.shape[0], 3*kernel_size*kernel_size).permute(0,2,1)
    output = F.fold(patches,
                    output_size=img_size,
                    kernel_size=kernel_size,
                    stride=dh)

    recovery_mask = F.fold(torch.ones_like(patches),
                    output_size=img_size,
                    kernel_size=kernel_size,
                    stride=dh)
    output /= recovery_mask
    x = output.squeeze(0).detach().cpu()
    x = transforms.ToPILImage()(x)
    frame_idx = int(image_file.split("\\")[-1][:-4][5:])
    x.save(f'output3/frame{frame_idx}.jpg')

model.train()
