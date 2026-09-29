#!/usr/bin/env python3
"""Derive paired uncertainty and post-hoc coverage diagnostics from completed runs."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from uji.metrics import conditional_intervals,paired_differences
from uji.data import load_data,XY
from uji.common import dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    d=load_data();rows=[]
    for sub,manfile in [('final','manifest.json'),('matched','matched_manifest.json')]:
        path=out/sub/'episode_metrics.csv'
        if not path.exists():continue
        frame=pd.read_csv(path)
        conditional_intervals(frame).to_csv(out/sub/'conditional_intervals.csv',index=False)
        paired_differences(frame).to_csv(out/sub/'paired_differences.csv',index=False)
        m=json.loads((out/manfile).read_text())
        for dom in m['test_domains']:
            spec=m['domains'][dom];q=d.loc[spec['query_rows']].drop_duplicates('gid')
            for es,order in spec['episodes'].items():
                for k in m['budgets']:
                    s=d.loc[order[:k]]
                    dy=q[XY].to_numpy()[:,None,:]-s[XY].to_numpy()[None,:,:]
                    oracle=np.sqrt((dy*dy).sum(-1)).min(-1)
                    rows.append({'stage':sub,'domain':dom,'support_seed':int(es),'K':k,
                        'labelled_position_fraction':k/spec['positions'],
                        'mean_nearest_support_xy_m':float(oracle.mean()),
                        'p90_nearest_support_xy_m':float(np.quantile(oracle,.9)),
                        'semantics':'POST-HOC coverage using hidden coordinates; never an algorithm score or training input'})
    pd.DataFrame(rows).to_csv(out/'support_coverage_diagnostic.csv',index=False)
    dump(out/'report_scope.json',{'conditional_intervals':'resample model/support seed axes; hold 3 floors and query sets fixed',
        'population_generalization_interval':False,'coverage_uses_hidden_labels_only_for_posthoc_diagnostics':True})
if __name__=='__main__':main()
