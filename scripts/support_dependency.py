#!/usr/bin/env python3
"""Development-only falsification: does changing support labels change predictions?

This script does not retrain or select hyperparameters. It can be run before final.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from uji.common import seed_all,device_name,dump
from uji.run import load_run
from uji.data import Transform,XY
from uji.models import make_model
from uji.engine import load_checkpoint,predict_cmanp
from uji.metrics import metrics
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
cfg,m,d=load_run(a.out);device=device_name(a.device);seed_all(17,cfg['runtime']['threads']);results=[]
for seed in cfg['training']['seeds']:
    payload=load_checkpoint(Path(a.out)/f'dev_cmanp_{seed}'/'best.pt',device)
    tf=Transform(**payload['transform']);model=make_model('cmanp',cfg).to(device);model.load_state_dict(payload['model'])
    for dom in m['dev_domains']:
        spec=m['domains'][dom];b=int(dom[1]);q=d.loc[spec['query_rows']]
        xq=torch.as_tensor(tf.x(q),device=device)[None]
        for k in m['budgets']:
            for es in cfg['evaluation']['dev_support_seeds']:
                c=d.loc[spec['episodes'][str(es)][:k]]
                xc=torch.as_tensor(tf.x(c),device=device)[None];yc=torch.as_tensor(tf.y(c),device=device)[None]
                normal,_=predict_cmanp(model,xc,yc,xq,cfg['evaluation']['query_chunk'])
                # Fixed cyclic label permutation has no tunable effect size or spatial oracle.
                corrupt,_=predict_cmanp(model,xc,yc.roll(1,dims=1),xq,cfg['evaluation']['query_chunk'])
                axy=tf.inverse(normal,b);bxy=tf.inverse(corrupt,b)
                ra,_=metrics(q[XY].to_numpy(),axy,q.gid.to_numpy());rb,_=metrics(q[XY].to_numpy(),bxy,q.gid.to_numpy())
                move,_=metrics(axy,bxy,q.gid.to_numpy())
                results.append({'domain':dom,'seed':seed,'support_seed':es,'K':k,
                    'correct_mde':ra['mde_m'],'permuted_mde':rb['mde_m'],'prediction_movement_m':move['mde_m']})
pd.DataFrame(results).to_csv(Path(a.out)/'support_dependency_dev.csv',index=False)
print(pd.DataFrame(results).to_string(index=False))
