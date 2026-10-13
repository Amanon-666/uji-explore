#!/usr/bin/env python3
"""Auditable UJI cross-building transfer-GP pilot; not a TNKN reproduction.

Only UCI trainingData.csv is used; validationData.csv stays untouched.
The "paper-like" setting matches published record budgets, not unpublished splits,
normalization or the TNKN neural kernel. Strict settings isolate physical positions.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

DATA_URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
ZIP_HASH = "893512b82dfd7a7c345d84195b1c8019fbca0fa0d7820ce491ce5aa45ec3782f"
WAPS = [f"WAP{i:03d}" for i in range(1, 521)]
XY = ["LONGITUDE", "LATITUDE"]
RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
NOISE = 0.15 ** 2
Y_SCALE = 50.0

def read_uji(cache: Path) -> pd.DataFrame:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(DATA_URL, timeout=120) as request:
            data = request.read()
        if hashlib.sha256(data).hexdigest() != ZIP_HASH:
            raise ValueError("Official UJI zip hash mismatch; refusing unverified data")
        cache.write_bytes(data)
    data = cache.read_bytes()
    if hashlib.sha256(data).hexdigest() != ZIP_HASH:
        raise ValueError("UJI zip checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        frame = pd.read_csv(archive.open("UJIndoorLoc/trainingData.csv"))
    if len(frame) != 19937 or list(frame.columns[:520]) != WAPS:
        raise ValueError("Unexpected trainingData CSV")
    frame["pos"] = (frame.LONGITUDE.round(1).astype(str)
                    + "|" + frame.LATITUDE.round(1).astype(str))
    frame["domain"] = "B" + frame.BUILDINGID.astype(str) + "F" + frame.FLOOR.astype(str)
    return frame

def choose_rows(df: pd.DataFrame, source_building: int, target_building: int,
                target_floor: int | None, budget: int, seed: int,
                paper_like: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if target_floor is None:
        src = df[df.BUILDINGID == source_building]
        tar = df[df.BUILDINGID == target_building]
    else:
        src = df[(df.BUILDINGID == target_building) & (df.FLOOR != target_floor)]
        tar = df[(df.BUILDINGID == target_building) & (df.FLOOR == target_floor)]
    source_ids = rng.choice(src.index.to_numpy(), size=min(1000,len(src)), replace=False)
    if paper_like:
        perm = rng.permutation(tar.index.to_numpy())
        if len(perm) <= budget + 1000:
            raise ValueError("Insufficient paper-like target rows")
        support_ids, query_ids = perm[:budget], perm[budget:budget + 1000]
    else:
        groups = tar.groupby("pos").indices
        positions = np.array(sorted(groups))
        rng.shuffle(positions)
        query_positions = set(positions[:max(1,int(np.ceil(.30*len(positions))))])
        support_positions = positions[max(1,int(np.ceil(.30*len(positions)))):]
        if len(support_positions) < budget:
            raise ValueError("Not enough distinct target locations")
        support_ids = []
        for p in support_positions[:budget]:
            index_in_tar = rng.choice(groups[p])
            support_ids.append(tar.index[index_in_tar])
        eligible = tar[tar.pos.isin(query_positions)].index.to_numpy()
        query_ids = rng.choice(eligible,size=min(1000,len(eligible)),replace=False)
    return df.loc[source_ids],df.loc[support_ids],df.loc[query_ids]

def normalized_rss(frame: pd.DataFrame, selection: np.ndarray) -> np.ndarray:
    raw = frame[WAPS].to_numpy(dtype=np.float64)[:, selection]
    return np.where(raw==100.0,0.0,(raw+110.0)/110.0)

def rbf(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    return np.exp(-0.5 * cdist(a,b,metric="sqeuclidean")/(length*length))

def transfer_gp(s: pd.DataFrame, t: pd.DataFrame, q: pd.DataFrame, mode: str):
    s_visible = (s[WAPS].to_numpy() != 100).any(axis=0)
    t_visible = (t[WAPS].to_numpy() != 100).any(axis=0)
    if mode == "all":
        selected = np.ones(520,dtype=bool)
    elif mode == "shared_support":
        selected = s_visible & t_visible
    else:
        raise ValueError(mode)
    count = int(selected.sum())
    if count < 1:
        return None
    xs,xt,xq = [normalized_rss(f, selected) for f in (s,t,q)]
    # Source-only length-scale heuristic: no target query features or labels.
    rng = np.random.default_rng(8334)
    ids = rng.choice(len(xs),min(300,len(xs)),replace=False)
    distances = cdist(xs[ids],xs[ids])
    positive = distances[distances>1e-8]
    length = max(float(np.median(positive)) if len(positive) else .1, .05)
    ys_origin = s[XY].to_numpy().mean(axis=0)
    yt_origin = t[XY].to_numpy().mean(axis=0)
    ys = (s[XY].to_numpy(dtype=float)-ys_origin)/Y_SCALE
    yt = (t[XY].to_numpy(dtype=float)-yt_origin)/Y_SCALE

    kss = rbf(xs,xs,length)
    kss.flat[::len(xs)+1] += NOISE
    fac_s=cho_factor(kss,lower=True,check_finite=False)
    ks_t=rbf(xs,xt,length)
    ks_q=rbf(xs,xq,length)
    kt_t=rbf(xt,xt,length)
    kt_t.flat[::len(xt)+1] += NOISE
    kt_q=rbf(xt,xq,length)
    alpha_s=cho_solve(fac_s,ys,check_finite=False)
    v_t=cho_solve(fac_s,ks_t,check_finite=False)
    v_q=cho_solve(fac_s,ks_q,check_finite=False)
    mu_t=ks_t.T @ alpha_s
    mu_q=ks_q.T @ alpha_s
    corr_tt=ks_t.T @ v_t
    corr_tq=ks_t.T @ v_q
    results={}
    lml={}
    for rho in RHO_GRID:
        covariance=kt_t-rho**2*corr_tt
        covariance=(covariance+covariance.T)*.5
        factor=cho_factor(covariance,lower=True,check_finite=False)
        innovation=yt-rho*mu_t
        alpha=cho_solve(factor,innovation,check_finite=False)
        pred=rho*mu_q+(kt_q-rho**2*corr_tq).T @ alpha
        results[rho]=pred*Y_SCALE+yt_origin
        # Conditional target-label evidence, no test labels are used for selection.
        logdet=2*np.log(np.diag(factor[0])).sum()
        lml[rho]=float(-.5*np.sum(innovation*alpha)-len(yt[0])*.5*logdet)
    chosen=max(RHO_GRID,key=lambda x:lml[x])
    actual=q[XY].to_numpy()
    metrics={}
    for name,rho in [("target_gp",0.0),("pooled_residual_gp",1.0),
                     ("transfer_evidence_gp",chosen)]:
        error=np.linalg.norm(results[rho]-actual,axis=1)
        metrics[name]={"mean":float(error.mean()),"median":float(np.median(error)),
                       "p90":float(np.quantile(error,.90))}
    return {"num_shared_visible_ap_from_support":int((s_visible&t_visible).sum()),
            "num_features":count,"length":length,"selected_rho":chosen,
            "target_label_evidence":{str(x):lml[x] for x in RHO_GRID},
            "metrics":metrics}

def evaluate(df, task, source, target, floor, budget, seed, paper_like, mode):
    s,t,q=choose_rows(df,source,target,floor,budget,seed,paper_like)
    answer=transfer_gp(s,t,q,mode)
    if answer is None:
        return None
    overlap=bool(set(t.pos)&set(q.pos))
    overlap_frac=float(q.pos.isin(set(t.pos)).mean())
    return {"task":task,"feature_mode":mode,"budget":budget,"seed":seed,
            "protocol":"paper_like_records" if paper_like else "strict_distinct_positions",
            "source_rows":len(s),"target_support_rows":len(t),
            "support_unique_positions":t.pos.nunique(),"query_rows":len(q),
            "query_unique_positions":q.pos.nunique(),
            "support_query_position_overlap_fraction":overlap_frac,
            "support_query_position_overlap":overlap,**answer}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("reports/cross_building_transfer"))
    p.add_argument("--data",type=Path,default=Path("data/raw/ujiindoorloc.zip"))
    p.add_argument("--quick",action="store_true",help="B1->B2 and one floor only; smoke/pilot")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    df=read_uji(args.data)
    cases=[
        ("B1_to_B2_paper_records",1,2,None,25,20261013,True,["all"]),
        ("B1_to_B2_strict",1,2,None,25,20261013,False,["all","shared_support"]),
        ("B0_to_B2_strict",0,2,None,25,20261013,False,["all","shared_support"]),
        ("B0_to_B1_strict",0,1,None,25,20261013,False,["all"]),
        ("B1_other_floors_to_F3",1,1,3,5,20261013,False,["all"]),
        ("B1_other_floors_to_F3",1,1,3,10,20261013,False,["all"]),
        ("B1_other_floors_to_F3",1,1,3,20,20261013,False,["all"]),
        ("B2_other_floors_to_F4",2,2,4,5,20261013,False,["all"]),
        ("B2_other_floors_to_F4",2,2,4,10,20261013,False,["all"]),
        ("B2_other_floors_to_F4",2,2,4,20,20261013,False,["all"]),
    ]
    if args.quick:
        cases=cases[:2]+cases[4:5]
    rows=[]
    for task,s,t,f,k,seed,paper,modes in cases:
        for mode in modes:
            row=evaluate(df,task,s,t,f,k,seed,paper,mode)
            if row is not None:
                rows.append(row)
                brief={key:row[key] for key in ("task","feature_mode","selected_rho","budget")}
                brief["errors"]={k:round(v["mean"],3) for k,v in row["metrics"].items()}
                print(json.dumps(brief),flush=True)
    with (args.out/"results.json").open("w") as f:
        json.dump({"profile":"quick_pilot" if args.quick else "single_seed_pilot",
                   "official_zip_sha256":ZIP_HASH,
                   "source":"UCI official trainingData.csv only",
                   "method":"fixed-RBF conditional two-task GP; rho selected by target-label evidence",
                   "not_reproduced":"AAMAS 2020 TNKN architecture, its preprocessing/normalization, and its scalar RMSE",
                   "target_label_policy":"only K target labels; unseen query labels only for error evaluation",
                   "output_coordinate_policy":"task-specific mean of observed labels; common fixed 50-unit scale",
                   "elapsed_seconds":round(time.monotonic()-start,2),
                   "rows":rows},f,indent=2,ensure_ascii=False)
    pd.DataFrame([{"task":r["task"],"mode":r["feature_mode"],"K":r["budget"],
                   "overlap_query_fraction":r["support_query_position_overlap_fraction"],
                   "shared_AP":r["num_shared_visible_ap_from_support"],
                   "selected_rho":r["selected_rho"],
                   **{name:score["mean"] for name,score in r["metrics"].items()}}
                  for r in rows]).to_csv(args.out/"summary.csv",index=False)
    print("EXPERIMENT_OK rows=",len(rows),"elapsed_seconds=",round(time.monotonic()-start,2),flush=True)

if __name__=="__main__":
    main()
