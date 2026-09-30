"""Fetch verified UCI data and audit the publicly released CNNLoc dev split."""
from __future__ import annotations
import argparse, hashlib, json, urllib.request, zipfile, shutil
from pathlib import Path
from collections import defaultdict,deque
import pandas as pd
from .core import load_csvs,write_json,digest

UCI='https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip'
ZIP_SHA='893512b82dfd7a7c345d84195b1c8019fbca0fa0d7820ce491ce5aa45ec3782f'
CNN_COMMIT='ec8d6001289a1239e21b23b87d52f16e1086d43f'
CNN_URL=f'https://raw.githubusercontent.com/XudongSong/CNNLoc/{CNN_COMMIT}/UJIIndoorLoc_codes/AllValuationData.csv'
CNN_SHA='de85e570ac24ca8df61e1b885cf354bd46ebb54128ca83d9be2230c90064e29b'


def fetch(url,path,sha):
    if not path.exists():
        tmp=path.with_suffix(path.suffix+'.tmp')
        with urllib.request.urlopen(url,timeout=90) as response,tmp.open('wb') as f:
            shutil.copyfileobj(response,f)
        if digest(tmp)!=sha:
            tmp.unlink();raise ValueError(f'Checksum mismatch for {url}')
        tmp.replace(path)
    if digest(path)!=sha:raise ValueError(f'Checksum mismatch: {path}')


def row_hash(df):
    # CSV floating-point serialization differs; tolerance is one micrometre,
    # far below RSSI label resolution. Every other one of 529 fields is matched.
    a=df.iloc[:,:529].astype('float64').copy()
    for k in ['LONGITUDE','LATITUDE']:a[k]=a[k].round(6)
    return pd.util.hash_pandas_object(a,index=False).values


def match_author_split(train,external,author):
    pool=defaultdict(deque)
    for rid,h in zip(train.rid,row_hash(train)):pool[int(h)].append(int(rid))
    dev=[]
    for h in row_hash(author):
        if not pool[int(h)]:raise ValueError('Author dev row not found in source training multiset')
        dev.append(pool[int(h)].popleft())
    overlap=len(set(map(int,row_hash(author))) & set(map(int,row_hash(external))))
    if overlap:raise ValueError('Author dev contains official external test rows')
    return dict(upstream='https://github.com/XudongSong/CNNLoc',commit=CNN_COMMIT,
      file='UJIIndoorLoc_codes/AllValuationData.csv',sha256=CNN_SHA,
      match='All 529 columns; coordinates rounded to 1e-6 meter for text-rounding tolerance',
      unmatched_rows=0,official_validation_matches=0,
      identical_original_rows=len(train)-len(set(map(int,row_hash(train)))),
      duplicate_policy='match multiplicities; equivalent identical rows assigned lowest unused rid',
      dev_ids=sorted(dev))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='data/raw/graph');p.add_argument('--zip',dest='zip_path')
    p.add_argument('--author-csv',help='Local file avoids download; SHA verified')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    archive=Path(args.zip_path) if args.zip_path else out/'ujiindoorloc.zip'
    fetch(UCI,archive,ZIP_SHA)
    with zipfile.ZipFile(archive) as z:
        for name in ['trainingData.csv','validationData.csv']:
            members=[n for n in z.namelist() if Path(n).name==name]
            if len(members)!=1:raise ValueError('Unexpected UCI archive layout')
            (out/name).write_bytes(z.read(members[0]))
    train,external=load_csvs(out)
    author=Path(args.author_csv) if args.author_csv else out/'CNNLoc-AllValuationData.csv'
    fetch(CNN_URL,author,CNN_SHA)
    recovered=match_author_split(train,external,pd.read_csv(author))
    tracked=Path(__file__).parent/'cnnloc_dev_ids.json'
    if tracked.exists() and json.loads(tracked.read_text())['dev_ids']!=recovered['dev_ids']:
        raise ValueError('Tracked split differs from pinned upstream recovery')
    # Audit is an output, never silently rewrites a committed protocol file.
    write_json(out/'cnnloc_split_audit.json',recovered)
    write_json(out/'dataset_audit.json',dict(training_n=len(train),external_n=len(external),
      train_floor_counts=train.groupby('FLOOR').size().to_dict(),external_floor_counts=external.groupby('FLOOR').size().to_dict(),
      pooled_floor_counts=pd.concat([train,external]).groupby('FLOOR').size().to_dict(),
      cnnloc_fit_n=len(train)-len(recovered['dev_ids']),cnnloc_dev_n=len(recovered['dev_ids']),
      csv_sha256={n:digest(out/n) for n in ['trainingData.csv','validationData.csv']}))
    print(f'Verified {len(train)} official training, {len(external)} external, {len(recovered["dev_ids"])} author dev rows.')

if __name__=='__main__':main()
