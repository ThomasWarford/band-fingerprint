import numpy as np
from fastai.vision.all import *
from skimage import io

def load_tiff_uint16_image(fn):
    "Open and load a uint16 as a float32 numpy array"
    arr = io.imread(fn).astype(np.float32)
    return arr

class TiffImage(PILBase):
    @classmethod
    def create(cls, fn:(Path,str), **kwargs)->None:
        "Open a uint16 tiff image from path `fn`"
        arr = load_tiff_uint16_image(fn)
        out = Image.fromarray(arr, "F")  
        return cls(out)
TiffImage._tensor_cls = TensorImageBW

class TensorImageNoised(TensorImageBW): pass
class TiffImageNoised(TiffImage): pass

TiffImageNoised._tensor_cls = TensorImageNoised

class AddNoiseTransform(Transform):
    "Add noise to image"
    order = 11
    def __init__(self, std=1, mean=0):self.std=std; self.mean=mean
    def encodes(self, o:TensorImageNoised): 
        return o + (self.std * torch.randn(*o.shape)+self.mean).to(o.device)
    
def cutout_gaussian(
    x:Tensor, # Input image 
    areas:list # List of areas to cutout. Order rl,rh,cl,ch
):
    "Replace all `areas` in `x` with N(0.5,0.25) noise"
    chan,img_h,img_w = x.shape[-3:]
    for rl,rh,cl,ch in areas: x[..., rl:rh, cl:ch].normal_(0.5, 0)
    return x
    
class RandomErasingTransform(RandomErasing):
    "Randomly selects a rectangle region in an image and randomizes its pixels."
    order = 100 # After Normalize
    def encodes(self,x:TensorImageNoised):
        count = random.randint(1, self.max_count)
        _,img_h,img_w = x.shape[-3:]
        area = img_h*img_w/count
        areas = [self._bounds(area, img_h, img_w) for _ in range(count)]
        return cutout_gaussian(x, areas)



# FOLLOWING KEPT FOR COMPATIBILITY ONLY:

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
    
