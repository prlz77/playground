from .conv4 import Conv4
import torch

def get_backbone(hparams):
    if hparams['backbone'] == "conv4":
        return Conv4(hparams)
    else:
        raise ValueError(hparams['backbone'])
