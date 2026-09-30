"""GConvLoc reconstruction and explicitly separate strict DG extensions."""
from __future__ import annotations
import argparse,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch,torch_geometric
from .core import GConvLoc,Preprocess,XY,load_csvs,source_split,graph_edges,metrics,write_json,digest


def partitions(train,external,protocol,target):
    if protocol=='standard':return train,external,None
    if protocol=='floor-dg':
        if target not in (0,1,2,3):raise ValueError('Target raw FLOOR 0..3 required')
        pool=train[train.FLOOR.isin([0,1,2,3])]
        return pool[pool.FLOOR!=target],pool[pool.FLOOR==target],external[external.FLOOR==target]
    if protocol=='building-dg':
        if target not in (0,1,2):raise ValueError('Target BUILDINGID 0..2 required')
        return train[train.BUILDINGID!=target],train[train.BUILDINGID==target],external[external.BUILDINGID==target]
    raise ValueError(protocol)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--protocol',choices=['standard','floor-dg','building-dg'],default='standard')
    p.add_argument('--target',type=int,default=0);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--split-seed',type=int,default=0);p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--patience',type=int,default=150);p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--dev-split',choices=['cnnloc','per-location'],default='cnnloc')
    p.add_argument('--dev-fraction',type=float,default=.1);p.add_argument('--k',type=int,default=23)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--activation',choices=['relu','elu'],default='relu')
    p.add_argument('--output',choices=['linear','sigmoid'],default='linear')
    p.add_argument('--backend',choices=['memory','pyg'],default='memory')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--phase',choices=['develop','test','all'],default='develop')
    p.add_argument('--allow-test',action='store_true')
    return p


def train_model(args,fit,dev,out):
    if (out/'FROZEN.json').exists():raise RuntimeError('Run already exists; choose empty output directory.')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    pre=Preprocess('gconvloc').fit(fit);xr,xd=pre.x(fit),pre.x(dev);t0=time.time()
    er=graph_edges(xr,k=args.k).to(args.device);ed=graph_edges(xr,xd,k=args.k).to(args.device)
    X=torch.tensor(xr,device=args.device);Y=torch.tensor(pre.y(fit),device=args.device)
    XD=torch.tensor(np.r_[xr,xd],device=args.device);YD=torch.tensor(pre.y(dev),device=args.device)
    model=GConvLoc(args.activation,args.output,args.backend).to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    frozen=dict(config=vars(args),fit_ids=fit.rid.astype(int).tolist(),dev_ids=dev.rid.astype(int).tolist(),
      preprocessing=pre.state(),paper_target_mde=7.59 if args.protocol=='standard' else None,
      status='independent reconstruction; missing author code and unspecified settings',
      software=dict(python=platform.python_version(),torch=torch.__version__,pyg=torch_geometric.__version__),
      source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(out/'FROZEN.json',frozen)
    import shutil
    (out/'source_snapshot').mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,out/'source_snapshot'/path.name)
    best=float('inf');best_epoch=0
    for epoch in range(1,args.epochs+1):
        model.train();opt.zero_grad(set_to_none=True);pred=model(X,er);loss=(pred-Y).square().mean()
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {epoch}')
        loss.backward();opt.step()
        if epoch%args.eval_every==0 or epoch==args.epochs or epoch==1:
            model.eval()
            with torch.no_grad():yp=model(XD,ed)[len(fit):];devloss=float((yp-YD).square().mean())
            m=metrics(pre.inverse(yp.cpu().numpy()),dev[XY].values,pre.yscale)
            row=dict(epoch=epoch,train_mse=float(loss.detach()),dev_mse=devloss,dev_mde_m=m['mde_m'],elapsed_seconds=time.time()-t0)
            with (out/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if devloss<best:
                best,best_epoch=devloss,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save(dict(state=state,preprocessing=pre.state(),config=vars(args),fit_ids=frozen['fit_ids'],selected_epoch=best_epoch),out/'best.pt')
            print(json.dumps(row),flush=True)
            if epoch-best_epoch>=args.patience:break
    summary=dict(selected_epoch=best_epoch,best_dev_mse=best,epochs_completed=epoch,seconds=time.time()-t0,
      parameters=sum(p.numel() for p in model.parameters()),reached_epoch_cap=epoch==args.epochs)
    write_json(out/'development.json',summary)


def evaluate(args,train,test,external,out):
    if not args.allow_test:raise RuntimeError('Evaluation needs --allow-test after development freeze.')
    if (out/'TEST_EVALUATED.json').exists():raise RuntimeError('Already evaluated; use saved predictions.')
    checkpoint=torch.load(out/'best.pt',map_location=args.device,weights_only=True);c=checkpoint['config']
    if c['protocol']!=args.protocol or c['target']!=args.target:raise ValueError('Checkpoint protocol mismatch')
    pre=Preprocess.restore(checkpoint['preprocessing']);fit=train[train.rid.isin(checkpoint['fit_ids'])]
    if len(fit)!=len(checkpoint['fit_ids']):raise ValueError('Reference rows not found')
    model=GConvLoc(c['activation'],c['output'],c['backend']).to(args.device)
    model.load_state_dict(checkpoint['state']);model.eval();xr=pre.x(fit);rows=[];predictions=[]
    write_json(out/'TEST_EVALUATED.json',dict(checkpoint_sha256=digest(out/'best.pt'),test_ids=test.rid.astype(int).tolist(),selected_epoch=checkpoint['selected_epoch']))
    for role,df in [('test',test),('external_stress',external)]:
        if df is None or len(df)==0:continue
        xq=pre.x(df);e=graph_edges(xr,xq,c['k']).to(args.device)
        with torch.no_grad():norm=model(torch.tensor(np.r_[xr,xq],device=args.device),e)[len(fit):].cpu().numpy()
        pred=pre.inverse(norm)
        rows.append(dict(role=role,building='all',floor='all',method='GConvLoc-reconstruction',**metrics(pred,df[XY].values,pre.yscale)))
        record=df[['rid','BUILDINGID','FLOOR',*XY,'PHONEID','USERID','TIMESTAMP']].copy()
        record['pred_x']=pred[:,0];record['pred_y']=pred[:,1];record['error_m']=np.linalg.norm(pred-df[XY].values,axis=1);record['role']=role
        predictions.append(record)
        for (b,f),g in record.groupby(['BUILDINGID','FLOOR']):
            rows.append(dict(role=role,building=int(b),floor=int(f),method='GConvLoc-reconstruction',**metrics(g[['pred_x','pred_y']].values,g[XY].values,pre.yscale)))
        from sklearn.neighbors import KNeighborsRegressor
        for k in [1,7]:
            knn=KNeighborsRegressor(n_neighbors=min(k,len(fit)),weights='uniform',metric='euclidean');knn.fit(xr,fit[XY].values)
            rows.append(dict(role=role,building='all',floor='all',method=f'{k}NN',**metrics(knn.predict(xq),df[XY].values,pre.yscale)))
    result=pd.DataFrame(rows);result.to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False,float_format='%.12g')
    print(result.to_string(index=False),flush=True)


def main():
    args=parser().parse_args()
    if args.epochs<1 or args.k<1 or args.patience<1 or not 0<args.dev_fraction<1:raise ValueError('Invalid budgets')
    torch.set_num_threads(args.threads);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train,external=load_csvs(args.data);source,test,stress=partitions(train,external,args.protocol,args.target)
    if args.phase in ['develop','all']:
        if args.dev_split=='cnnloc':
            audit=Path(args.data)/'cnnloc_split_audit.json'
            if not audit.exists():raise FileNotFoundError('Run python -m graph_repro.prepare first')
            manifest=json.loads(audit.read_text())
            if manifest['sha256']!='de85e570ac24ca8df61e1b885cf354bd46ebb54128ca83d9be2230c90064e29b':raise ValueError('Unexpected author split hash')
            ids=manifest['dev_ids']
            if len(ids)!=2132 or len(set(ids))!=2132:raise ValueError('Invalid author split IDs')
            dev=source[source.rid.isin(ids)].copy();fit=source[~source.rid.isin(ids)].copy()
            if len(dev)==0 or len(fit)==0:raise ValueError('CNNLoc split empty in this task')
        else:fit,dev=source_split(source,args.split_seed,args.dev_fraction)
        train_model(args,fit,dev,out)
    if args.phase in ['test','all']:evaluate(args,train,test,stress,out)

if __name__=='__main__':main()
