import numpy as np
from fastai.vision.all import *
from skimage import io
from torchvision.transforms import RandomErasing

def load_tiff_uint32_image(fn):
    "Open and load a raw image with rawpy"
    image = io.imread(fn).astype(np.int32)
    return image


class Tiff32Image(PILBase):
    @classmethod
    def create(cls, fn:(Path,str), **kwargs)->None:
        "Open a raw image from path `fn`"
        arr = load_tiff_uint32_image(fn)
        out = Image.fromarray(arr, "I")             
        return cls(out)

Tiff32Image._tensor_cls = TensorImageBW
    
class TensorImageNoised(TensorImageBW): pass
class Tiff32ImageNoised(Tiff32Image): pass

Tiff32ImageNoised._tensor_cls = TensorImageNoised

class AddNoiseTransform(Transform):
    "Add noise to image"
    order = 11
    def __init__(self, std=1000, mean=0):self.std=std; self.mean=mean
    def encodes(self, o:TensorImageNoised): 
        return o + (self.std * torch.randn(*o.shape)+self.mean).to(o.device)
    
class RandomEraseTransform(Transform):
    "Add noise to image"
    order = 11
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0,):
        self.p=p
        self.scale=scale
        self.ratio=ratio
        self.value=value
                 
    def encodes(self, o:TensorImageNoised): 
        x, y, h, w, v = RandomErasing.get_params(o, self.scale, self.ratio, self.value)
        return F.erase(o, x, y, h, w, v)