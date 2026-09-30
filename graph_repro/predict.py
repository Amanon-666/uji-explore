"""Predict coordinates for arbitrary RSSI-only queries; no query labels required."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .core import GConvLoc,Preprocess,WAPS,load_csvs,graph_edges

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--query',required=True,help='CSV with WAP001..WAP520; coordinates optional and ignored')
    p.add_argument('--out',required=True);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    t,_=load_csvs(a.data);ref=t[t.rid.isin(c['fit_ids'])]
    if len(ref)!=len(c['fit_ids']):raise ValueError('Checkpoint reference rows missing')
    query=pd.read_csv(a.query)
    if not set(WAPS).issubset(query.columns):raise ValueError('520 WAP columns required')
    if len(query)==0:raise ValueError('Query CSV is empty')
    x=query[WAPS].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.all((x==100)|((x>=-104)&(x<=0))):raise ValueError('Invalid RSSI values')
    pre=Preprocess.restore(c['preprocessing']);config=c['config']
    xr,xq=pre.x(ref),pre.x(query);edges=graph_edges(xr,xq,config['k'])
    net=GConvLoc(config['activation'],config['output'],config['backend']);net.load_state_dict(c['state']);net.eval()
    with torch.no_grad():y=pre.inverse(net(torch.tensor(np.r_[xr,xq]),edges)[len(ref):].numpy())
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(y,columns=['pred_LONGITUDE','pred_LATITUDE']).to_csv(a.out,index=False)

if __name__=='__main__':main()
