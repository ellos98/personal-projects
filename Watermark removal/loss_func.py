import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGG19(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x]) 
        for x in range(2,7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7,12):
            self.slice3.add_module(str(x), vgg[x])            
        for x in range(12,21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21,30):
            self.slice5.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]
    
class VGGLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.vgg = VGG19().eval().to(device=device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss