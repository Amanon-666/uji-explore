from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from .common import upstream_imports


def make_model(kind, cfg, dim_x=520, dim_y=2):
    if kind == 'cmanp':
        upstream_imports()
        from models.cmanp import CMANP
        args = dict(cfg['model']); args.update(dim_x=dim_x, dim_y=dim_y)
        return CMANP(**args)
    if kind == 'mlp':
        widths = [dim_x] + list(cfg['mlp']['hidden']) + [dim_y]
        layers=[]
        for i, (a,b) in enumerate(zip(widths[:-1],widths[1:])):
            layers.append(nn.Linear(a,b))
            if i < len(widths)-2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    raise ValueError(kind)


def squared_dist(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    return np.maximum(((a[:,None,:]-b[None,:,:])**2).sum(-1),0.)


def wknn(xc,yc,xq,k):
    """Euclidean RSS distance, inverse-distance weighting, exact-match handling."""
    dist=np.sqrt(squared_dist(xq,xc))
    ix=np.argsort(dist,axis=1,kind='stable')[:,:min(int(k),len(xc))]
    nearest=np.take_along_axis(dist,ix,axis=1)
    exact=nearest==0
    weights=1./np.maximum(nearest,np.finfo(np.float64).eps)
    has_exact=exact.any(1)
    weights[has_exact]=exact[has_exact].astype(float)
    weights/=weights.sum(1,keepdims=True)
    return (yc[ix]*weights[...,None]).sum(1)


def rbf_regression(xc,yc,xq,length_multiplier,ridge):
    """Support-only kernel ridge with support-mean prior and median distance bandwidth."""
    dcc=squared_dist(xc,xc)
    positive=dcc[dcc>0]
    length2=float(np.median(positive)) if len(positive) else 1.
    length2*=float(length_multiplier)**2
    kernel=np.exp(-dcc/(2.*length2))
    center=np.asarray(yc,dtype=np.float64).mean(0)
    weights=np.linalg.solve(kernel+float(ridge)*np.eye(len(xc)),yc-center)
    return center+np.exp(-squared_dist(xq,xc)/(2.*length2))@weights


def adapt_mlp(base,xc,yc,lr,steps,device):
    """Exactly one full-support Adam update per step; optimizer resets per episode."""
    model=copy.deepcopy(base).to(device).train()
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss=(model(xc)-yc).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite MLP adaptation loss')
        loss.backward(); opt.step()
    return model.eval()
