from typing import Any, Optional, Dict, List, Union
import torchvision.transforms as t
from PIL.Image import Image
import torch


IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

class BYOLTransforms1:
    def __init__(
            self,
            input_size: int = 224,
            min_scale: float = 0.08,
    ):
        self.transforms = t.Compose([
            t.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            t.RandomHorizontalFlip(),
            t.RandomApply([
                t.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            t.RandomGrayscale(p=0.2),
            t.ToTensor(),
            t.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"])
        ])

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transforms(x)
    

class BYOLTransforms:
    def __init__(
            self,
            view1: Optional[BYOLTransforms1] = None,
            view2: Optional[BYOLTransforms1] = None,
    ):  
        self.view1 = view1 or BYOLTransforms1()
        self.view2 = view2 or BYOLTransforms1()

    def __call__(self, image: Union[torch.Tensor, Image]) -> Union[List[torch.Tensor], List[Image]]:
        return [self.view1(image), self.view2(image)]

        

    
