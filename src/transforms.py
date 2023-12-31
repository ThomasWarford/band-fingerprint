#!/usr/bin/env python
"""
Transforms for images applied using fastai dataloaders.
"""

from fastai import *
from fastai.vision.all import *

class Binarize(Transform):
    def __init__(self, threshold=0.8):self.threshold=threshold
    def encodes(self, o): 
        mask = (o > (255 * self.threshold))
        return mask * 255