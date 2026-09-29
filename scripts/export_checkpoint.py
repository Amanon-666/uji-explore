#!/usr/bin/env python3
"""Export a TRUSTED training checkpoint to tensor-only inference payload."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from uji.engine import load_checkpoint
from uji.common import file_hash,dump
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
s=load_checkpoint(a.input,'cpu')
light={k:s[k] for k in ['model','transform','config','kind','seed','step']}
light['status']='pilot_only' if s['config']['run_profile']!='full' else 'full_run_checkpoint'
light['source_checkpoint_sha256']=file_hash(a.input)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
torch.save(light,out)
dump(out.with_suffix('.json'),{k:v for k,v in light.items() if k!='model'})
print(str(out),file_hash(out))
