#!/usr/bin/env python3
"""Diagnostic for the B1->B2 fixed-RBF GP result, not a TNKN reconstruction.

For each support/query split, inspect raw kernel blocks and the source-only
conditional prediction. No Query labels are used to fit or choose parameters.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.cross_building_transfer import (read_uji,choose_rows,normalized_rss,rbf,
                                                 WAPS,XY,NOISE,Y_SCALE)

def describe(df,src_id,tgt_id,floor,K,seed):
    s,t,q=choose_rows(df,src_id,tgt_id,floor,K,seed,False)
    Xs,Xt,Xq=[normalized_rss(v,np.ones(520,dtype=bool)) for v in (s,t,q)]
    rng=np.random.default_rng(8334)
    sample=rng.choice(len(Xs),min(300,len(Xs)),replace=False)
    ds=cdist(Xs[sample],Xs[sample])
    ell=max(float(np.median(ds[ds>1e-8])),.05)
    kst=rbf(Xs,Xt,ell)
    ktt=rbf(Xt,Xt,ell)
    ks_q=rbf(Xs,Xq,ell)
    kt_q=rbf(Xt,Xq,ell)
    d_st=cdist(Xs,Xt)
    d_tt=cdist(Xt,Xt)
    source_visible=(s[WAPS].to_numpy()!=100).any(axis=0)
    target_visible=(t[WAPS].to_numpy()!=100).any(axis=0)
    ys=(s[XY].to_numpy()-s[XY].to_numpy().mean(axis=0))/Y_SCALE
    yt=(t[XY].to_numpy()-t[XY].to_numpy().mean(axis=0))/Y_SCALE
    kss=rbf(Xs,Xs,ell)
    kss.flat[::len(Xs)+1]+=NOISE
    fac=cho_factor(kss,lower=True,check_finite=False)
    pred_s_q=ks_q.T@cho_solve(fac,ys,check_finite=False)
    pred_s_t=kst.T@cho_solve(fac,ys,check_finite=False)
    # This is the uncorrected source's prediction in coordinates centered at
    # the observed target support centroid, explicitly not test-trained.
    t_center=t[XY].to_numpy().mean(axis=0)
    source_only_error=np.linalg.norm(Y_SCALE*pred_s_q+t_center-q[XY].to_numpy(),axis=1).mean()
    # Diagnose an omitted degree of freedom of the pilot: signed rho in [-1,1].
    # The AAMAS TNKN kernel allows signed cross-domain coefficients.
    signed=None
    if floor is None:
        import experiments.cross_building_transfer as gp_module
        old_grid=gp_module.RHO_GRID
        gp_module.RHO_GRID=(-1.0,-.75,-.5,-.25,0.0,.25,.5,.75,1.0)
        try:
            signed=gp_module.transfer_gp(s,t,q,"all")
        finally:
            gp_module.RHO_GRID=old_grid
    target_centroid_error=np.linalg.norm(t_center-q[XY].to_numpy(),axis=1).mean()
    source_support_error=np.linalg.norm(pred_s_t-yt,axis=1).mean()*Y_SCALE
    target_support_centroid_error=np.linalg.norm(yt,axis=1).mean()*Y_SCALE
    # Check labels/coordinates of a *building-wide* group that can contain
    # several physical floors in the same planar location.
    groups=q.groupby("pos").FLOOR.nunique()
    target_building=df[df.BUILDINGID==tgt_id].copy()
    # Independently audit reused *planar* coordinates across floors.
    target_building["planar_pos"]=(target_building.LONGITUDE.round(1).astype(str)
                                  +"|"+target_building.LATITUDE.round(1).astype(str))
    all_groups=target_building.groupby("planar_pos").FLOOR.nunique()
    samefloor_mode=floor is not None
    ss=np.median(ds[ds>1e-8])
    return {"source":src_id,"target":tgt_id,"target_floor":floor,"K":K,
            "seed":seed,"length":ell,"source_target_overlap_AP":int((source_visible&target_visible).sum()),
            "source_only_AP":int((source_visible&~target_visible).sum()),
            "target_only_AP":int((~source_visible&target_visible).sum()),
            "source_target_RSS_distance_median":float(np.median(d_st)),
            "source_source_RSS_distance_median":float(ss),
            "target_target_RSS_distance_median":float(np.median(d_tt[d_tt>1e-8])),
            "cross_kernel_mean":float(kst.mean()),
            "cross_kernel_median":float(np.median(kst)),
            "cross_kernel_gt_0p5_fraction":float((kst>.5).mean()),
            "cross_kernel_gt_0p1_fraction":float((kst>.1).mean()),
            "target_kernel_off_diagonal_median":float(np.median(ktt[~np.eye(len(t),dtype=bool)])),
            "source_only_query_MDE":float(source_only_error),
            "signed_rho_target_evidence_selected":signed["selected_rho"] if signed is not None else None,
            "signed_rho_query_oracle_selected":signed["oracle_best_rho_diagnostic_only"] if signed is not None else None,
            "signed_rho_evidence_selected_MDE":signed["metrics"]["transfer_evidence_gp"]["mean"] if signed is not None else None,
            "signed_rho_query_oracle_MDE":min(signed["rho_query_error_diagnostic_only"].values()) if signed is not None else None,
            "centroid_only_query_MDE":float(target_centroid_error),
            "source_only_support_MDE":float(source_support_error),
            "centroid_only_support_MDE":float(target_support_centroid_error),
            "source_predicted_deviation_query_rms_m":float(Y_SCALE*np.sqrt((pred_s_q**2).sum(axis=1).mean())),
            "target_support_deviation_rms_m":float(Y_SCALE*np.sqrt((yt**2).sum(axis=1).mean())),
            "source_coord_span_m":(s[XY].max()-s[XY].min()).tolist(),
            "target_coord_span_m":(t[XY].max()-t[XY].min()).tolist(),
            "target_building_planar_groups_shared_by_multiple_floors":int((all_groups>1).sum()) if not samefloor_mode else None,
            "query_planar_groups_shared_by_multiple_floors":int((groups>1).sum()) if not samefloor_mode else None}

def main():
    d=read_uji(ROOT/"data/raw/ujiindoorloc.zip")
    tasks=[("B1_to_B2",1,2,None,25),("B0_to_B2",0,2,None,25),
           ("B1_to_F3",1,1,3,5),("B1_to_F3",1,1,3,10),
           ("B2_to_F4",2,2,4,5)]
    records=[]
    for seed in range(20261013,20261018):
        for name,s,t,f,k in tasks:
            record=describe(d,s,t,f,k,seed)
            record["task"]=name
            records.append(record)
            print(json.dumps({key:record[key] for key in
                ("task","seed","cross_kernel_median","target_kernel_off_diagonal_median",
                 "source_only_query_MDE","centroid_only_query_MDE")}),flush=True)
    out=ROOT/"reports/cross_building_transfer"
    out.mkdir(parents=True,exist_ok=True)
    (out/"kernel_diagnostics.json").write_text(json.dumps(records,indent=2))
    print("DIAGNOSTICS_OK",len(records),flush=True)

if __name__=="__main__":
    main()
