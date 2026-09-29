from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .common import ROOT,dump,digest,file_hash,verify_upstream,environment,seed_all,device_name,source_fingerprint
from .data import load_data,make_protocol,validate_protocol,audit,Transform,domain_rows
from .models import make_model
from .engine import train,evaluate,load_checkpoint,tune_baselines
from .metrics import summarize


def config(path,profile):
    cfg=yaml.safe_load(Path(path).read_text())
    cfg['run_profile']=profile
    if profile in ('smoke','pilot'):
        cfg['training'].update(seeds=[0],max_steps=8 if profile=='smoke' else 300,
            batch_tasks=2,eval_every=4 if profile=='smoke' else 100,patience_checks=10)
        cfg['protocol']['support_seeds']=[0,1]
        cfg['evaluation']['dev_support_seeds']=[0]
        # Architecture unchanged. These are execution checks, not paper/full-benchmark results.
    return cfg


def prepare(args):
    cfg=config(args.config,args.profile);cfg['code_hash']=source_fingerprint();d=load_data();out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        raise FileExistsError('Run directory already initialized. Use a new directory instead of overwriting manifests.')
    dump(out/'config.json',cfg);audit(d,out/'audit')
    for matched in (False,True):
        m=make_protocol(d,cfg,matched);dump(out/('matched_manifest.json' if matched else 'manifest.json'),m)
    dump(out/'environment.json',environment());verify_upstream()
    print(f'Prepared and validated manifests in {out}')


def load_run(out):
    out=Path(out)
    cfg=json.loads((out/'config.json').read_text());m=json.loads((out/'manifest.json').read_text())
    if cfg.get('code_hash') is not None and cfg['code_hash']!=source_fingerprint():
        raise ValueError('Training code changed after protocol preparation. Start a new versioned run.')
    d=load_data();validate_protocol(d,m);verify_upstream()
    return cfg,m,d


def develop(args):
    out=Path(args.out);cfg,m,d=load_run(out);device=device_name(args.device)
    if (out/'TEST_OPENED.json').exists() or (out/'MATCHED_TEST_OPENED.json').exists():
        raise RuntimeError('Development is sealed after final test access. Do not tune on opened target results.')
    seed_all(0,cfg['runtime']['threads'])
    scaler=Transform.fit(domain_rows(d,m,m['fit_domains']))
    bc=tune_baselines(d,m,cfg,scaler,out)
    selections=[]
    for kind in ['cmanp','mlp']:
        for seed in cfg['training']['seeds']:
            path,best=train(d,m,cfg,kind,seed,out/f'dev_{kind}_{seed}',device,
                m['fit_domains'],m['dev_domains'],resume=args.resume)
            selections.append({'kind':kind,'seed':seed,'step':best['step'],'dev_score':best['score'],
                'checkpoint_sha256':file_hash(path)})
    # Median selected step across independent training seeds; all final seeds use the same budget.
    steps={kind:int(np.median([s['step'] for s in selections if s['kind']==kind])) for kind in ['cmanp','mlp']}
    freeze={'version':'1.0','config_hash':digest(cfg),'protocol_hash':m['hash'],
        'data_sha256':m['data_zip_sha256'],'upstream_commit':verify_upstream(),
        'selected_steps':steps,'baseline_config':bc,'selection_records':selections,
        'selection_domains':m['dev_domains'],'test_domains_used_for_selection':[],
        'profile':cfg['run_profile']}
    freeze['hash']=digest(freeze);dump(out/'FROZEN.json',freeze)
    print('Development completed. FROZEN.json contains all choices before target evaluation.')


def frozen(out,cfg,m):
    f=json.loads((Path(out)/'FROZEN.json').read_text())
    if f['config_hash']!=digest(cfg) or f['protocol_hash']!=m['hash'] or digest({k:v for k,v in f.items() if k!='hash'})!=f['hash']:
        raise ValueError('Frozen decision/config/protocol mismatch')
    return f


def final(args):
    out=Path(args.out);cfg,m,d=load_run(out);f=frozen(out,cfg,m);device=device_name(args.device)
    if not args.allow_test:
        raise ValueError('Final target evaluation needs --allow-test after development choices are frozen.')
    flag=out/('MATCHED_TEST_OPENED.json' if args.matched else 'TEST_OPENED.json')
    if flag.exists() and not args.resume:
        raise FileExistsError('Test results already opened. New development must use a new registered protocol version, not tune on these targets.')
    # Permission marker is an audit trail, not a cryptographic access-control claim.
    dump(flag,{'freeze_hash':f['hash'],'intent':'single frozen final evaluation','profile':cfg['run_profile']})
    if args.matched:
        m=json.loads((out/'matched_manifest.json').read_text());validate_protocol(d,m)
    finalout=out/('matched' if args.matched else 'final');finalout.mkdir(exist_ok=True)
    all_frames=[]
    for kind in ['cmanp','mlp']:
        for seed in cfg['training']['seeds']:
            cp=finalout/f'{kind}_{seed}'/'best.pt'
            if not (args.resume and cp.exists()):
                cp,_=train(d,m,cfg,kind,seed,finalout/f'{kind}_{seed}',device,
                    m['source_domains'],steps=f['selected_steps'][kind],resume=args.resume)
            payload=load_checkpoint(cp,device)
            tf=Transform(**payload['transform']);model=make_model(kind,cfg).to(device)
            model.load_state_dict(payload['model'])
            for evalkind in (['cmanp'] if kind=='cmanp' else ['mlp','mlp_ft']):
                frame,preds=evaluate(d,m,tf,cfg,m['test_domains'],evalkind,model,device,
                    stage='matched' if args.matched else 'primary',seed=seed,
                    baseline_cfg=f['baseline_config'],save_predictions=True)
                all_frames.append(frame)
                preds.to_csv(finalout/f'predictions_{evalkind}_{seed}.csv.gz',index=False,compression='gzip')
            if not args.matched:
                frame,_=evaluate(d,m,tf,cfg,m['test_domains'],kind,model,device,
                    stage='external_unseen_position',seed=seed,baseline_cfg=f['baseline_config'],external=True)
                all_frames.append(frame)
    # Deterministic support-only baselines run once per support set, not duplicated for model seeds.
    tf=Transform.fit(domain_rows(d,m,m['source_domains']))
    for kind in ['wknn','rbf','support_mean']:
        for ext in ([False] if args.matched else [False,True]):
            frame,preds=evaluate(d,m,tf,cfg,m['test_domains'],kind,device=device,
                stage='matched' if args.matched else ('external_unseen_position' if ext else 'primary'),
                seed=-1,baseline_cfg=f['baseline_config'],external=ext,save_predictions=True)
            all_frames.append(frame)
            preds.to_csv(finalout/f'predictions_{kind}_{ext}.csv.gz',index=False,compression='gzip')
    combined=pd.concat(all_frames,ignore_index=True)
    combined['profile']=cfg['run_profile'];combined['freeze_hash']=f['hash']
    combined.to_csv(finalout/'episode_metrics.csv',index=False)
    floors,macro=summarize(combined);floors.to_csv(finalout/'floor_metrics.csv',index=False);macro.to_csv(finalout/'macro_metrics.csv',index=False)
    dump(finalout/'completion.json',{'profile':cfg['run_profile'],'freeze_hash':f['hash'],
        'protocol_hash':m['hash'],'status':'complete','original_paper_numerical_reproduction':False})
    print(macro[['stage','model','K','adapt_steps','mde_m']].to_string(index=False))


def main():
    p=argparse.ArgumentParser(description='Cross-floor few-shot UJI benchmark')
    p.add_argument('command',choices=['prepare','develop','final'])
    p.add_argument('--config',default=str(ROOT/'configs/uji.yaml'))
    p.add_argument('--profile',choices=['full','pilot','smoke'],default='full')
    p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--resume',action='store_true');p.add_argument('--allow-test',action='store_true');p.add_argument('--matched',action='store_true')
    a=p.parse_args()
    {'prepare':prepare,'develop':develop,'final':final}[a.command](a)

if __name__=='__main__':main()
