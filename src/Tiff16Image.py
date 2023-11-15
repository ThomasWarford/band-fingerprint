import numpy as np
from fastai.vision.all import *
from skimage import io

def load_tiff_uint16_image(fn):
    "Open and load a raw image with rawpy"
    return io.imread(fn)


class Tiff16Image(PILBase):
    @classmethod
    def create(cls, fn:(Path,str), **kwargs)->None:
        "Open a raw image from path `fn`"
        arr = load_tiff_uint16_image(fn)
        return cls(Image.fromarray(arr.astype(np.float64)))
    
Tiff16Image._tensor_cls = TensorImageBW


# class PILImageNoised(PILImageBW): pass
# class TensorImageNoised(TensorImageBW): pass
# PILImageNoised._tensor_cls = TensorImageNoised

# class AddNoiseTransform(Transform):
#     "Add noise to image"
#     order = 11
#     def __init__(self, std=0.05, mean=-0.2):self.std=std; self.mean=mean
#     def encodes(self, o:TensorImageNoised): 
#         return o + (self.std * torch.randn(*o.shape)+self.mean).to(o.device)