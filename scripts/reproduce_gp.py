#!/usr/bin/env python3
"""Official CMANP/GP numerical-reproduction runner with explicit code/paper discrepancy.

GPU suggested for full 5-seed x 100k-step run. No UJI data are used here.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import yaml
from uji.common import ROOT,upstream_imports,verify_upstream,seed_all,dump,environment,device_name
from uji.models import make_model
upstream_imports()
from data.gp import GPSampler,RBFKernel,Matern52Kernel,PeriodicKernel

class PaperKernel:
    """Paper §4.2 length U[.6,1), vs official source U[.1,.6); expose, never hide."""
    def __init__(self,kind):self.kind=kind
    def __call__(self,x):
        import math
        length=.6+.4*torch.rand([x.shape[0],1,1,1],device=x.device)
        scale=.1+.9*torch.rand([x.shape[0],1,1],device=x.device)
        delta=(x.unsqueeze(-2)-x.unsqueeze(-3))/length
        if self.kind=='rbf': base=torch.exp(-.5*delta.square().sum(-1))
        elif self.kind=='matern':
            r=delta.norm(dim=-1)
            base=(1+math.sqrt(5)*r+5*r.square()/3)*torch.exp(-math.sqrt(5)*r)
        else:raise ValueError('Paper kernel sensitivity supports RBF/Matern only.')
        return scale.square()*base+.02**2*torch.eye(x.shape[-2],device=x.device)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]);p.add_argument('--eval-batches',type=int,default=3000)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);p.add_argument('--kernel-spec',choices=['code','paper'],default='code')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.steps<1 or a.eval_batches<1:raise ValueError('Positive steps/eval batches required')
    device=device_name(a.device);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'configs/uji.yaml').read_text());verify_upstream()
    targets={'rbf':1.24,'matern':.80}
    results=[]
    settings={'steps':a.steps,'seeds':a.seeds,'eval_batches':a.eval_batches,'kernel_spec':a.kernel_spec,
        'batch_size':16,'max_num_points':50,'lr':.0005,'weight_decay':0.,'model':cfg['model'],
        'upstream_commit':verify_upstream(),'environment':environment(),
        'paper_code_mismatch':'paper length uniform [0.6,1.0); code uniform [0.1,0.6)'}
    settings_file=out/'settings.json'
    if settings_file.exists():
        previous=json.loads(settings_file.read_text())
        for key in ['steps','seeds','eval_batches','kernel_spec','model']:
            if previous[key]!=settings[key]:raise ValueError('Resume settings mismatch')
    dump(settings_file,settings)
    for seed in a.seeds:
        seed_all(seed);model=make_model('cmanp',cfg,1,1).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=.0005)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100000)
        cp=out/f'seed_{seed}.pt';start=0
        if cp.exists():
            if not a.resume:raise FileExistsError(cp)
            state=torch.load(cp,map_location=device,weights_only=False)
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            start=state['step'];torch.set_rng_state(state['torch_rng'].cpu())
            if device=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        sampler=GPSampler(RBFKernel() if a.kernel_spec=='code' else PaperKernel('rbf'))
        begin=time.perf_counter()
        for step in range(start+1,a.steps+1):
            model.train();batch=sampler.sample(batch_size=16,max_num_points=50,device=device)
            opt.zero_grad(set_to_none=True);loss=model(batch).loss
            if not torch.isfinite(loss):raise FloatingPointError('GP training diverged')
            loss.backward();opt.step();scheduler.step()
            if step%1000==0 or step==a.steps:
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                    'step':step,'seed':seed,'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None},cp)
                event={'seed':seed,'step':step,'last_batch_loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-begin}
                with open(out/'train.jsonl','a') as f:f.write(json.dumps(event)+'\n')
                print(json.dumps(event),flush=True)
        model.eval()
        kernels={'rbf':RBFKernel(),'matern':Matern52Kernel(),'periodic':PeriodicKernel()} if a.kernel_spec=='code' else {'rbf':PaperKernel('rbf'),'matern':PaperKernel('matern')}
        for i,(name,kernel) in enumerate(kernels.items()):
            # Same test draw seed across model seeds, paired comparisons; independent of training stream.
            torch.manual_seed(91000+i)
            if device=='cuda':torch.cuda.manual_seed_all(91000+i)
            scores=[]
            sampler=GPSampler(kernel)
            with torch.no_grad():
                for _ in range(a.eval_batches):
                    batch=sampler.sample(batch_size=16,max_num_points=50,device=device)
                    scores.append(float(model(batch).tar_ll))
            results.append({'seed':seed,'kernel':name,'log_likelihood':float(np.mean(scores)),
                'paper_target':targets.get(name),'steps':a.steps,'eval_batches':a.eval_batches})
        dump(out/'seed_results.json',results)
    means={k:float(np.mean([r['log_likelihood'] for r in results if r['kernel']==k])) for k in kernels}
    dump(out/'summary.json',{'profile':'full_budget' if a.steps==100000 and len(a.seeds)==5 and a.eval_batches==3000 else 'smoke_or_pilot',
        'mean_log_likelihood':means,'paper_targets':targets,
        'difference_to_table4':{k:means[k]-v for k,v in targets.items()},
        'near_table4_0p1_rule':all(abs(means[k]-v)<=.1 for k,v in targets.items()),
        'rule_status':'project diagnostic tolerance, not author-defined acceptance test',
        'paper_code_kernel_discrepancy_unresolved':True})
    print(json.dumps(means,indent=2))
if __name__=='__main__':main()
