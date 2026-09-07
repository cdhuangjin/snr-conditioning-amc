"""Parameter-free, deterministic reflection padding for new Phase 7 protocols.

The supported domain is the nonnegative 1D padding used by frozen AWN lifting.
Forward values implement the same reflection index map. Backward uses slice,
flip and concatenation autograd, avoiding CUDA reflection-pad atomic scatter.
Floating-point gradient summation order can differ from native CPU padding.
"""
from numbers import Integral
import torch
from torch import nn

ADAPTER_ID='reflection_pad1d_slice_flip_cat_v1'


class DeterministicReflectionPad1d(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,Integral):padding=(int(padding),int(padding))
        if not isinstance(padding,(tuple,list)) or len(padding)!=2 or any(not isinstance(p,Integral) or p<0 for p in padding):
            raise ValueError('requires nonnegative integer left/right padding')
        self.padding=tuple(int(p) for p in padding)

    def forward(self,x):
        left,right=self.padding
        if x.ndim not in (2,3) or max(left,right)>=x.shape[-1]:
            raise ValueError('expected 2D/3D input with padding smaller than time dimension')
        return torch.cat((x[...,1:left+1].flip(-1),x,x[...,-right-1:-1].flip(-1)),dim=-1)

    def extra_repr(self):
        return str(self.padding)


def install_adapter(model):
    """Replace parameter-free child pads in-place; return exact replaced paths.

    Do not call on frozen baseline inference paths. The owning new protocol
    must record ADAPTER_ID, this file's hash and the returned module paths.
    State-dict keys/values are unchanged, allowing strict checkpoint loading.
    """
    replaced=[]
    def visit(module,prefix):
        for name,child in list(module.named_children()):
            path=f'{prefix}.{name}' if prefix else name
            if isinstance(child,nn.ReflectionPad1d):
                pad=DeterministicReflectionPad1d(child.padding)
                pad.train(child.training);setattr(module,name,pad);replaced.append(path)
            else:visit(child,path)
    visit(model,'')
    return replaced
