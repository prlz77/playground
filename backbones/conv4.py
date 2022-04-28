import torch
import torch.nn.functional as F
from backbones.heads import get_head
from backbones.utils import LayerNorm


class NonLinearity:
  @staticmethod
  def get(hparams):
    name = hparams.get('nonlinearity', 'SiLU')
    ret = getattr(torch.nn, name)(inplace=True)
    return ret


class Conv4(torch.nn.Module):
  def __init__(self, hparams):
    super().__init__()
    self.layer0 = torch.nn.Conv2d(hparams['conv4']['in_ch'], 32, 3, bias=False, padding=1, stride=2) # 28 -> 14
    self.ln0 = torch.nn.GroupNorm(32, 32)
    self.nonlinear0 = NonLinearity.get(hparams)
    self.layer1 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False) # 14->7 
    self.ln1 = torch.nn.GroupNorm(32, 64)
    self.nonlinear1 = NonLinearity.get(hparams)
    self.layer2 = torch.nn.Conv2d(64, 128, 3, padding=1, bias=False) # 7 -> 7
    self.ln2 = torch.nn.GroupNorm(32, 128)
    self.nonlinear2 = NonLinearity.get(hparams)
    self.layer3 = torch.nn.Conv2d(128, 256, 3, padding=1, bias=False) # 7 -> 7
    self.ln3 = torch.nn.GroupNorm(32, 256)
    self.nonlinear3 = NonLinearity.get(hparams)
    self.output_head = get_head(256, hparams["output_dim"], hparams)

  def forward(self, x):
    x = self.layer0(x)
    x = self.ln0(x)
    x = self.nonlinear0(x)
    x = self.layer1(x)
    x = self.ln1(x)
    x = self.nonlinear1(x)
    x = self.layer2(x)
    x = self.ln2(x)
    x = self.nonlinear2(x)
    x = self.layer3(x)
    x = self.ln3(x)
    features = self.nonlinear3(x)
    
    return features, self.output_head(features)