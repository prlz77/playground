import torch
import torch.nn.functional as F


def get_head(ni, no, hparams):
    if hparams['head'] == "linear":
        return Linear(ni, no, bias=False)
    elif hparams["head"] == "tokenizer":
        return TokenizerHead(ni, no, hparams)
    else:
        raise ValueError(hparams['backbone'])

class Linear(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x.mean((2,3)))

class TokenizerHead(torch.nn.Module):
    def __init__(self, ni, no, hparams):
        super().__init__()
        self.layer = torch.nn.Linear(ni, no, bias=False)
    
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        x = self.layer(x)
        return F.gumbel_softmax(x, dim=1, hard=True).view(b, h, w, -1).permute(0, 3, 1, 2)