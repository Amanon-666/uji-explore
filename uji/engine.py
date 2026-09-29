from __future__ import annotations
import copy, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import dump, digest, file_hash, seed_all, stable_seed
from .data import Transform, domain_rows, XY
from .models import make_model, wknn, rbf_regression, adapt_mlp
from .metrics import metrics

class EpisodeSampler:
    """Uniform floors, uniform distinct positions, one uniformly selected raw scan."""
    def __init__(self,d,tf,seed):
        self.rng=np.random.default_rng(seed)
        self.x=tf.x(d); self.y=tf.y(d)
        local=d.reset_index(drop=True)
        self.by_domain=[]
        for dom,g in local.groupby('domain'):
            self.by_domain.append([v.index.to_numpy() for _,v in g.groupby('gid')])

    def sample(self,batch_size,k,n_query,device):
        c=[];q=[]
        for _ in range(batch_size):
            groups=self.by_domain[int(self.rng.integers(len(self.by_domain)))]
            if len(groups)<k+n_query:
                raise ValueError(f'Episode requires {k+n_query} distinct positions, available {len(groups)}')
            positions=self.rng.choice(len(groups),k+n_query,replace=False)
            ids=[int(self.rng.choice(groups[j])) for j in positions]
            c.append(ids[:k]);q.append(ids[k:])
        c=np.asarray(c);q=np.asarray(q)
        return tuple(torch.as_tensor(a,device=device) for a in (self.x[c],self.y[c],self.x[q],self.y[q]))


def checkpoint(path,model,optimizer,scheduler,sampler,tf,cfg,manifest,kind,seed,step,best):
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'sampler_rng':sampler.rng.bit_generator.state,
        'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
        'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'transform':tf.as_dict(),'config':cfg,'config_hash':digest(cfg),'protocol_hash':manifest['hash'],
        'kind':kind,'seed':seed,'step':step,'best':best}
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def load_checkpoint(path,device='cpu'):
    # Only load checkpoints produced by this project or otherwise trusted by the user.
    return torch.load(path,map_location=device,weights_only=False)


def predict_cmanp(model,xc,yc,xq,chunk):
    from types import SimpleNamespace
    model.eval()
    with torch.no_grad():
        context=model.get_context_encoding(SimpleNamespace(xc=xc,yc=yc))
        means=[];stds=[]
        for i in range(0,xq.shape[1],chunk):
            p=model.predict(xc,yc,xq[:,i:i+chunk],context_encodings=context)
            means.append(p.mean[0].cpu().numpy());stds.append(p.stddev[0].cpu().numpy())
    return np.concatenate(means),np.concatenate(stds)


def evaluate(d,m,tf,cfg,domains,kind,model=None,device='cpu',stage='dev',seed=0,
             baseline_cfg=None,external=False,save_predictions=False):
    records=[];predictions=[]
    support_seeds=cfg['protocol']['support_seeds']
    baseline_cfg=baseline_cfg or cfg['baselines']
    for dom in domains:
        spec=m['domains'][dom];b=int(dom[1])
        if external:
            # Fixed query set across all label budgets: exclude all exact target-train positions.
            known=set(d.loc[spec['all_rows'],'gid'])
            q=d[(d.split=='external')&(d.domain==dom)&(~d.gid.isin(known))]
        else:
            q=d.loc[spec['query_rows']]
        if q.empty:
            raise ValueError(f'No query points in {dom}')
        xq=tf.x(q); yq=q[XY].to_numpy(dtype=np.float64); gids=q.gid.to_numpy()
        xqt=torch.as_tensor(xq,device=device)
        for es in support_seeds:
            order=spec['episodes'][str(es)]
            for k in m['budgets']:
                s=d.loc[order[:k]]
                if set(s.gid)&set(gids):
                    raise ValueError('Support-query exact coordinate overlap')
                xc=tf.x(s);yc=tf.y(s);xct=torch.as_tensor(xc,device=device);yct=torch.as_tensor(yc,device=device)
                steps_grid=baseline_cfg['adapt_steps'] if kind=='mlp_ft' else [0]
                for adapt_steps in steps_grid:
                    start=time.perf_counter();std=None
                    if kind=='cmanp':
                        yn,std=predict_cmanp(model,xct[None],yct[None],xqt[None],cfg['evaluation']['query_chunk'])
                    elif kind=='wknn':
                        yn=wknn(xc,yc,xq,baseline_cfg['wknn_k'])
                    elif kind=='rbf':
                        yn=rbf_regression(xc,yc,xq,baseline_cfg['rbf_length_multiplier'],baseline_cfg['rbf_ridge'])
                    elif kind in ('mlp','mlp_ft'):
                        active=adapt_mlp(model,xct,yct,baseline_cfg['ft_lr'],adapt_steps,device) if kind=='mlp_ft' else model.eval()
                        with torch.no_grad(): yn=active(xqt).cpu().numpy()
                    elif kind=='support_mean':
                        yn=np.broadcast_to(yc.mean(0),(len(xq),2))
                    else:
                        raise ValueError(kind)
                    if device=='cuda': torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    pred=tf.inverse(yn,b)
                    result,errors=metrics(yq,pred,gids,std,tf.buildings[str(b)]['scale'])
                    record={'stage':stage,'model':kind,'domain':dom,'K':k,'support_seed':es,
                        'model_seed':seed,'adapt_steps':adapt_steps,
                        'labels_consumed':0 if kind=='mlp' or (kind=='mlp_ft' and adapt_steps==0) else k,
                        'predict_and_adapt_seconds':elapsed,**result}
                    records.append(record)
                    if save_predictions:
                        for rid,y,p,error in zip(q.rid,yq,pred,errors):
                            predictions.append({**{z:record[z] for z in ['stage','model','domain','K','support_seed','model_seed','adapt_steps']},
                                'query_row':int(rid),'true_x':float(y[0]),'true_y':float(y[1]),
                                'pred_x':float(p[0]),'pred_y':float(p[1]),'error_m':float(error)})
    return pd.DataFrame(records),pd.DataFrame(predictions)


def train(d,m,cfg,kind,seed,out,device,fit_domains,dev_domains=None,steps=None,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    last=out/'last.pt'; best_path=out/'best.pt'
    if last.exists() and not resume:
        raise FileExistsError(f'{last} exists. Use --resume or choose a fresh run directory.')
    seed_all(seed,cfg['runtime']['threads'])
    if set(fit_domains) & set(m['test_domains']):
        raise ValueError('Target domain supplied to source training')
    source=domain_rows(d,m,fit_domains)
    tf=Transform.fit(source)
    model=make_model(kind,cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    total=steps or cfg['training']['max_steps']
    # The same 100k horizon is retained during refit, even when refitting fewer selected steps.
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['training']['max_steps'])
    sampler=EpisodeSampler(source,tf,stable_seed(seed,'training'))
    best={'score':1e100,'step':0,'stale':0};start_step=0
    if resume:
        state=load_checkpoint(last,device)
        if state['config_hash']!=digest(cfg) or state['protocol_hash']!=m['hash'] or state['transform']!=tf.as_dict():
            raise ValueError('Resume config/protocol/scaler mismatch')
        if state['kind']!=kind or state['seed']!=seed:
            raise ValueError('Resume model identity mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        sampler.rng.bit_generator.state=state['sampler_rng'];np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if device=='cuda' and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_step=state['step'];best=state['best']
    log_path=out/'learning_curve.jsonl'
    begin=time.perf_counter()
    for step in range(start_step+1,total+1):
        model.train()
        k=int(sampler.rng.choice(m['budgets']))
        xc,yc,xq,yq=sampler.sample(cfg['training']['batch_tasks'],k,cfg['training']['query_positions'],device)
        opt.zero_grad(set_to_none=True)
        if kind=='cmanp':
            distribution=model.predict(xc,yc,xq)
            loss=-distribution.log_prob(yq).sum(-1).mean()
        else:
            loss=(model(xq)-yq).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}; no automatic patching or test-driven retries.')
        loss.backward();opt.step();sched.step()
        check=(step%cfg['training']['eval_every']==0 or step==total)
        if check:
            event={'step':step,'train_loss_last_batch':float(loss.detach()),'lr':sched.get_last_lr()[0],
                   'elapsed_seconds':time.perf_counter()-begin,'seed':seed,'kind':kind}
            if dev_domains:
                vc=copy.deepcopy(cfg)
                vc['protocol']['support_seeds']=cfg['evaluation']['dev_support_seeds']
                frame,_=evaluate(d,m,tf,vc,dev_domains,kind,model,device)
                score=float(frame.groupby('domain').mde_m.mean().mean())
                event['dev_position_floor_macro_mde']=score
                if score<best['score']:
                    best={'score':score,'step':step,'stale':0}
                    checkpoint(best_path,model,opt,sched,sampler,tf,cfg,m,kind,seed,step,best)
                else: best['stale']+=1
            else:
                best={'score':0.,'step':step,'stale':0}
            checkpoint(last,model,opt,sched,sampler,tf,cfg,m,kind,seed,step,best)
            with open(log_path,'a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
            if dev_domains and best['stale']>=cfg['training']['patience_checks']:
                break
    if not dev_domains:
        # Final refit has no target-based checkpoint selection.
        import shutil
        shutil.copy2(last,best_path)
    return best_path,best


def tune_baselines(d,m,cfg,tf,out):
    """A small preregistered grid evaluated on development floors only."""
    vc=copy.deepcopy(cfg);vc['protocol']['support_seeds']=cfg['evaluation']['dev_support_seeds']
    rows=[];best={}
    grid=[('wknn',{'wknn_k':k}) for k in cfg['search']['wknn_k']]
    grid += [('rbf',{'rbf_length_multiplier':length,'rbf_ridge':ridge})
        for length in cfg['search']['rbf_length_multiplier'] for ridge in cfg['search']['rbf_ridge']]
    for kind,params in grid:
        bc={**cfg['baselines'],**params}
        f,_=evaluate(d,m,tf,vc,m['dev_domains'],kind,baseline_cfg=bc)
        score=float(f.groupby('domain').mde_m.mean().mean())
        rows.append({'model':kind,'params':params,'dev_mde':score})
        if kind not in best or score<best[kind]['dev_mde']:
            best[kind]={'params':params,'dev_mde':score}
    selected={**cfg['baselines'],**best['wknn']['params'],**best['rbf']['params']}
    dump(Path(out)/'baseline_search.json',{'all_trials':rows,'selected':selected,'selection_domains':m['dev_domains']})
    return selected
