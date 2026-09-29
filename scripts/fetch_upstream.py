#!/usr/bin/env python3
"""Obtain the pinned PUBLIC upstream repository; fail closed on byte differences."""
from __future__ import annotations
import json,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uji.common import ROOT,verify_upstream
manifest=json.loads((ROOT/'third_party/manifest.json').read_text())
dest=ROOT/'third_party/constant-memory-anp'
if not dest.exists():
    with tempfile.TemporaryDirectory(prefix='cmanp-') as temp:
        tmp=Path(temp)/'upstream'
        subprocess.run(['git','clone','--no-checkout','--filter=blob:none',manifest['repository'],str(tmp)],check=True)
        subprocess.run(['git','-C',str(tmp),'checkout','--detach',manifest['commit']],check=True)
        shutil.rmtree(tmp/'.git')
        shutil.copytree(tmp,dest)
print('Verified upstream commit',verify_upstream())
