import numpy as np
from fastai.vision.all import *
from skimage import io

def load_tiff_uint32_image(fn):
    "Open and load a raw image with rawpy"
    return io.imread(fn, np.uint32)


class Tiff32Image(PILBase):
    @classmethod
    def create(cls, fn:(Path,str), **kwargs)->None:
        "Open a raw image from path `fn`"
        arr = load_tiff_uint32_image(fn)
        return cls(Image.fromarray(arr.astype(np.float64)))
    
Tiff32Image._tensor_cls = TensorImageBW