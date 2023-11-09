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
    def __init__(self, n_latent=128):
        super().__init__()
        self.encode = nn.Sequential(
            conv(1, 4), # 224x128
            conv(4, 8), # 112x64
            conv(8, 16), # 66x32
            conv(16, 32), # 33x16
            # nn.Flatten(),
            # nn.Linear(32*33*16, n_latent),
            # nn.Tanh()
        )
        
        self.decode_linear=nn.Sequential(
            nn.Linear(n_latent, 32*33*16),
            nn.ReLU()
        )
        
        self.decode = nn.Sequential(
            deconv(32, 16), # 32x32
            deconv(16, 8), # 64x64
            deconv(8, 4), # 128x128
            deconv(4, 1, act=False), # 256x256
            nn.Sigmoid()
        )
        
    
    def forward(self, x):
        output = self.encode(x)
        # output = self.decode_linear(output)
        # output = output.view(-1, 32, 33, 16)
        return self.decode(output)