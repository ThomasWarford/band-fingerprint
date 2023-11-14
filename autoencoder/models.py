from fastai import *
from fastai.vision.all import *

def conv(
    ni, # input channels (3 for rgb image)
    nf, # output channels
    ks=3, # kernal size (ks * ks)
    act=True, #whether we add an activation layer
    norm=False): 

    layers = [nn.Conv2d(ni, nf, ks, stride=2, padding=ks//2, bias=False)]
    if norm: layers.append(nn.BatchNorm2d(nf))
    if act: layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)

def deconv(
    ni, # input channels (3 for rgb image)
    nf, # output channels
    ks=5, # kernal size (ks * ks)
    act=True,
    norm=False): #whether we add an activation layer)
    
    layers = [nn.UpsamplingNearest2d(scale_factor=2), 
             nn.Conv2d(ni, nf, ks, stride=1, padding=ks//2, bias=False),]
    
    if norm: layers.append(nn.BatchNorm2d(nf))
    if act: layers.append(nn.ReLU())
    
    
    return nn.Sequential(*layers)



class SimpleAutoencoder(nn.Module):
    def __init__(self, n_latent=128, norm=False):
        super().__init__()
        self.encode = nn.Sequential(
            conv(1, 4, norm=norm), # 6x64
            conv(4, 8, norm=norm), # 3x32
            nn.Flatten(),
            nn.Linear(3*32*8, n_latent),
            nn.Tanh(),
        )
        
        self.decode_linear=nn.Sequential(
            nn.Linear(n_latent, 8*3*32),
            nn.BatchNorm1d(8*3*32),
            nn.ReLU(),
        )
        
        self.decode = nn.Sequential(
            deconv(8, 4, norm=norm), # 6x64
            deconv(4, 1, norm=norm, act=False), # 12x128
            nn.Sigmoid()
        )
        
    
    def forward(self, x):
        out = self.encode(x)
        out = self.decode_linear(out)
        out = out.view(-1, 8, 3, 32)
        return self.decode(out)