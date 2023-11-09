import numpy as np
from fastai.vision.all import *
from skimage import io

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