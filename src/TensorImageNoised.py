#!/usr/bin/env python
"""
TensorImageNoised class, and corresponding transforms.

This class exists so autoencoder inputs can be subjected to augmentation transforms without the outputs being affected when using fastai dataloaders.
"""

from fastai import *
from fastai.vision.all import *

class PILImageNoised(PILImage): pass
class TensorImageNoised(TensorImage): pass
PILImageNoised._tensor_cls = TensorImageNoised

class AddNoiseTransform(Transform):
    "Add noise to image"
    split_idx=0
    order = 11
    def __init__(self, noise_factor=0.3): self.noise_factor = noise_factor
    def encodes(self, o:TensorImageNoised): return o + (self.noise_factor * torch.randn(*o.shape).to(o.device))

class RandomErasingTransform(RandTransform):
    "Randomly selects a rectangle region in an image and randomizes its pixels."
    order = 100 # After Normalize
    def __init__(self, 
        p:float=0.5, # Probability of appying Random Erasing
        sl:float=0., # Minimum proportion of erased area
        sh:float=0.3, # Maximum proportion of erased area
        min_aspect:float=0.3, # Minimum aspect ratio of erased area
        max_count:int=1 # Maximum number of erasing blocks per image, area per box is scaled by count
    ):
        store_attr()
        super().__init__(p=p)
        self.log_ratio = (math.log(min_aspect), math.log(1/min_aspect))

    def _bounds(self, area, img_h, img_w):
        r_area = random.uniform(self.sl,self.sh) * area
        aspect = math.exp(random.uniform(*self.log_ratio))
        return self._slice(r_area*aspect, img_h) + self._slice(r_area/aspect, img_w)
    
    def _slice(self, area, sz):
        bound = int(round(math.sqrt(area)))
        loc = random.randint(0, max(sz-bound, 0))
        return loc,loc+bound
    def cutout_gaussian(self,
        x:TensorImageNoised, # Input image 
        areas:list # List of areas to cutout. Order rl,rh,cl,ch
    ):
        "Replace all `areas` in `x` with N(0,1) noise"
        chan,img_h,img_w = x.shape[-3:]
        for rl,rh,cl,ch in areas: x[..., rl:rh, cl:ch].normal_()
        return x

    def encodes(self,x:TensorImageNoised):
        count = random.randint(1, self.max_count)
        _,img_h,img_w = x.shape[-3:]
        area = img_h*img_w/count
        areas = [self._bounds(area, img_h, img_w) for _ in range(count)]
        return self.cutout_gaussian(x, areas)