#!/usr/bin/env python3
"""Predict XY from 520 WAP columns, a labelled support CSV, and exported checkpoint.

Queries do NOT need coordinates, phone/user/time/floor/room labels.
The known building selects a source-fitted coordinate frame.
"""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from uji.common import device_name,seed_all
from uji.data import WAPS,XY,Transform
from uji.models import make_model
from uji.engine import predict_cmanp

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--support',required=True)
p.add_argument('--query',required=True);p.add_argument('--building',type=int,required=True)
p.add_argument('--output',required=True);p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);a=p.parse_args()
seed_all(0);device=device_name(a.device)
s=torch.load(a.checkpoint,map_location=device,weights_only=True)
if s['kind']!='cmanp':raise ValueError('This command expects an exported CMANP checkpoint')
cfg=s['config'];model=make_model('cmanp',cfg).to(device);model.load_state_dict(s['model'])
tf=Transform(**s['transform']);c=pd.read_csv(a.support);q=pd.read_csv(a.query)
if len(c)==0 or len(q)==0:raise ValueError('Support/query must be nonempty')
for d in [c,q]:
    if not set(WAPS)<=set(d):raise ValueError('CSV must include WAP001...WAP520')
    x=d[WAPS].to_numpy()
    if not (((x>=-104)&(x<=0))|(x==100)).all():raise ValueError('Unexpected UJI RSSI encoding')
if not set(XY)<=set(c):raise ValueError('Support requires LONGITUDE/LATITUDE labels')
c['BUILDINGID']=a.building
xc=torch.as_tensor(tf.x(c),device=device)[None];yc=torch.as_tensor(tf.y(c),device=device)[None]
xq=torch.as_tensor(tf.x(q),device=device)[None]
mu,sd=predict_cmanp(model,xc,yc,xq,cfg['evaluation']['query_chunk'])
xy=tf.inverse(mu,a.building);scale=tf.buildings[str(a.building)]['scale']
pd.DataFrame({'pred_x':xy[:,0],'pred_y':xy[:,1],'std_x':sd[:,0]*scale,'std_y':sd[:,1]*scale}).to_csv(a.output,index=False)
print(f'{len(q)} predictions written to {a.output}; checkpoint status={s["status"]}')
