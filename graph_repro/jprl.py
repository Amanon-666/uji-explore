"""Independent JPRL Eqs 6-11 regression reconstruction on strict train-only floors.

Official validation remains external. This is NOT a numerical reproduction of
Table 12's apparently pooled data. Model selection never uses target labels.
"""
from __future__ import annotations
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .core import JPRLNet,Preprocess,jprl_l2,load_csvs,metrics,write_json,XY,digest


def fit_candidate(args,fit,dev,pre,lam,out):
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    x=torch.tensor(pre.x(fit),device=args.device);y=torch.tensor(pre.y(fit),device=args.device)
    domain=torch.tensor(fit.FLOOR.values,device=args.device)
    xd=torch.tensor(pre.x(dev),device=args.device)
    members=[np.flatnonzero(fit.FLOOR.values==d) for d in sorted(fit.FLOOR.unique())]
    model=JPRLNet().to(args.device);opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    best=float('inf');step_best=0;t0=time.time()
    path=out/f'lambda_{lam:g}';path.mkdir()
    for step in range(1,args.steps+1):
        ids=np.concatenate([rng.choice(v,args.batch_per_domain,replace=True) for v in members])
        pred,h=model(x[ids]);regression=(pred-y[ids]).square().sum(1).mean()
        alignment=jprl_l2(h,y[ids],domain[ids],args.epsilon) if lam else h.sum()*0
        loss=regression+lam*alignment
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite lambda={lam}, step={step}')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%args.eval_every==0 or step in [1,args.steps]:
            with torch.no_grad(): yp=model(xd)[0].cpu().numpy()
            val=metrics(pre.inverse(yp),dev[XY].values,pre.yscale)
            row=dict(step=step,regression=float(regression.detach()),alignment=float(alignment.detach()),
                     dev_sum_mae=val['normalized_sum_mae'],dev_mde_m=val['mde_m'],seconds=time.time()-t0)
            with (path/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if val['normalized_sum_mae']<best:
                best,step_best=val['normalized_sum_mae'],step
                torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                       preprocessing=pre.state(),lambda_value=lam,step=step),path/'best.pt')
            if step==args.steps or step%1000==0:print(f'target {args.target} lambda {lam}: {json.dumps(row)}',flush=True)
            if step-step_best>=args.patience:break
    return dict(lambda_value=lam,best_dev_sum_mae=best,selected_step=step_best,completed_steps=step,seconds=time.time()-t0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--target',type=int,choices=[0,1,2,3],required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--batch-per-domain',type=int,default=16)
    p.add_argument('--epsilon',type=float,default=.001);p.add_argument('--patience',type=int,default=1000)
    p.add_argument('--eval-every',type=int,default=100);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--phase',choices=['develop','test','all'],default='develop');p.add_argument('--allow-test',action='store_true')
    args=p.parse_args();torch.set_num_threads(args.threads)
    if min(args.steps,args.batch_per_domain,args.eval_every,args.patience)<1 or args.epsilon<=0:raise ValueError('Positive budgets required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train,ext=load_csvs(args.data);pool=train[train.FLOOR.isin([0,1,2,3])]
    source=pool[pool.FLOOR!=args.target];target=pool[pool.FLOOR==args.target]
    if args.phase in ['develop','all']:
        if (out/'FROZEN.json').exists():raise RuntimeError('Choose an empty run directory.')
        rng=np.random.default_rng(0);fit_ids=[];dev_ids=[]
        for _,g in source.groupby('FLOOR',sort=True):
            ids=rng.permutation(g.index.values);n=max(1,round(len(ids)*.1))
            dev_ids.extend(ids[:n]);fit_ids.extend(ids[n:])
        fit=source.loc[sorted(fit_ids)];dev=source.loc[sorted(dev_ids)];pre=Preprocess('jprl').fit(fit)
        write_json(out/'FROZEN.json',dict(config=vars(args),fit_ids=fit.rid.tolist(),dev_ids=dev.rid.tolist(),
          preprocessing=pre.state(),protocol='strict-floor-dg',missing_value='retain raw 100 then source z-score',
          model='520-ReLU260-Sigmoid2',paper_lambda_grid=[.001,.1,10.,1000.],
          source_sha256={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
          software=dict(python=platform.python_version(),torch=torch.__version__)))
        import shutil
        (out/'source_snapshot').mkdir()
        for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source_snapshot'/f.name)
        candidates=[fit_candidate(args,fit,dev,pre,lam,out) for lam in [0.,.001,.1,10.,1000.]]
        selected=min(candidates[1:],key=lambda v:v['best_dev_sum_mae'])
        write_json(out/'selection.json',dict(candidates=candidates,selected=selected))
    if args.phase in ['test','all']:
        if not args.allow_test:raise RuntimeError('Freeze all choices before --allow-test.')
        if (out/'TEST_EVALUATED.json').exists():raise RuntimeError('Already tested; use saved results.')
        frozen=json.loads((out/'FROZEN.json').read_text())
        if frozen['config']['target']!=args.target:raise ValueError('Target differs from frozen run')
        selection=json.loads((out/'selection.json').read_text());lam=selection['selected']['lambda_value']
        write_json(out/'TEST_EVALUATED.json',dict(target=args.target,lambda_value=lam,source_ids=source.rid.tolist(),target_ids=target.rid.tolist()))
        results=[];prediction_frames=[]
        for name,value in [('JPRL-reconstruction',lam),('ERM-same-backbone',0.)]:
            checkpoint=torch.load(out/f'lambda_{value:g}'/'best.pt',map_location=args.device,weights_only=True)
            pre=Preprocess.restore(checkpoint['preprocessing']);model=JPRLNet().to(args.device)
            model.load_state_dict(checkpoint['state']);model.eval()
            for role,df in [('target_train',target),('external_stress',ext[ext.FLOOR==args.target])]:
                with torch.no_grad():yp=model(torch.tensor(pre.x(df),device=args.device))[0].cpu().numpy()
                pred=pre.inverse(yp)
                results.append(dict(method=name,role=role,raw_floor=args.target,seed=args.seed,lambda_value=value,
                  step=checkpoint['step'],**metrics(pred,df[XY].values,pre.yscale)))
                frame=df[['rid','BUILDINGID','FLOOR',*XY,'PHONEID','USERID','TIMESTAMP']].copy()
                frame['pred_x']=pred[:,0];frame['pred_y']=pred[:,1];frame['role']=role;frame['method']=name
                prediction_frames.append(frame)
        pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
        pd.concat(prediction_frames).to_csv(out/'predictions.csv',index=False,float_format='%.12g')
        print(pd.DataFrame(results).to_string(index=False),flush=True)

if __name__=='__main__':main()
