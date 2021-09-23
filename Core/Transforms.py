import numpy as np

from torchvision.transforms import transforms
from torchvision.transforms import functional as F

class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
        super().__init__(size, scale, ratio)
        
    def __call__(self, img):
        if self.initialize_flag:
            self.random_value = self.get_params(img, self.scale, self.ratio)
            self.initialize_flag = False

        return F.resized_crop(img, *self.random_value, self.size, self.interpolation)

    def initialize(self):
        self.initialize_flag = True

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img):
        if self.random_value > self.p:
            return F.hflip(img)
        return img
    
    def initialize(self):
        self.random_value = np.random.random()

class ToTensor(transforms.ToTensor): pass

class Normalize(transforms.Normalize): pass

class Resize(transforms.Resize): pass

class CenterCrop(transforms.CenterCrop): pass

class Compose(transforms.Compose):
    def initialize(self):
        for transform in self.transforms:
            if "initialize" in dir(transform):
                transform.initialize()