from __future__ import annotations
import numpy as np
import pandas as pd


def weighted_quantile(x,w,p):
    order=np.argsort(x,kind='stable'); x=np.asarray(x)[order]; w=np.asarray(w)[order]
    return float(x[min(np.searchsorted(np.cumsum(w)/w.sum(),p,side='left'),len(x)-1)])


def metrics(y,pred,gids,std=None,scale=1.):
    """Each unique coordinate gets equal total weight, however many scans it has."""
    error=np.linalg.norm(np.asarray(y,dtype=np.float64)-np.asarray(pred,dtype=np.float64),axis=-1)
    _, inverse, counts=np.unique(gids,return_inverse=True,return_counts=True)
    w=1./counts[inverse]; w/=w.sum()
    result={'mde_m':float(w@error),'sample_mde_m':float(error.mean()),
        'median_m':weighted_quantile(error,w,.5),'p90_m':weighted_quantile(error,w,.9),
        'rmse_2d_m':float(np.sqrt(w@(error**2))),
        'within_5m':float(w@(error<=5)),'within_10m':float(w@(error<=10)),
        'query_scans':len(error),'query_positions':len(counts)}
    if std is not None:
        sigma=np.asarray(std,dtype=np.float64)*scale
        if not np.isfinite(sigma).all() or not (sigma>0).all():
            raise FloatingPointError('Invalid predictive standard deviation')
        residual=np.asarray(y)-np.asarray(pred)
        nll=(.5*(residual/sigma)**2+np.log(sigma)+.5*np.log(2*np.pi)).sum(-1)
        mahal=((residual/sigma)**2).sum(-1)
        result.update(nll_planar=float(w@nll),ellipse95_coverage=float(w@(mahal<=5.991464547107979)))
    return result,error


def summarize(frame):
    """No scan-level pseudoreplication; report variation conditional on fixed floors."""
    metrics_cols=[c for c in ['mde_m','sample_mde_m','median_m','p90_m','rmse_2d_m',
                  'within_5m','within_10m','nll_planar','ellipse95_coverage'] if c in frame]
    key=[c for c in ['stage','model','K','adapt_steps'] if c in frame]
    floor=frame.groupby(key+['domain'],dropna=False)[metrics_cols].mean().reset_index()
    macro=floor.groupby(key,dropna=False)[metrics_cols].mean().reset_index()
    return floor,macro


def conditional_intervals(frame,repeats=2000,seed=20260929):
    """Crossed seed bootstrap conditional on the THREE fixed floors/query sets.

    Not a confidence interval for all buildings/floors. Baseline model_seed=-1
    has no training-randomness dimension. Pilot intervals are descriptive only.
    """
    rng=np.random.default_rng(seed);rows=[]
    for key,g in frame.groupby(['stage','model','K','adapt_steps']):
        values=g.groupby(['model_seed','support_seed']).mde_m.mean().unstack('support_seed').to_numpy()
        if not np.isfinite(values).all():raise ValueError('Missing crossed seed cells')
        boot=[]
        for _ in range(repeats):
            ix=rng.integers(len(values),size=len(values));jx=rng.integers(values.shape[1],size=values.shape[1])
            boot.append(float(values[ix][:,jx].mean()))
        rows.append(dict(zip(['stage','model','K','adapt_steps'],key),mean_mde=float(values.mean()),
            lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),
            training_repetitions=len(values),support_repetitions=values.shape[1],
            interpretation='conditional on fixed floors/query positions; no floor-population inference'))
    return pd.DataFrame(rows)


def paired_differences(frame,repeats=2000,seed=20260929):
    """Paired CMANP minus support-only baseline MDE; negative favors CMANP."""
    rng=np.random.default_rng(seed);rows=[]
    for (stage,k),g in frame.groupby(['stage','K']):
        a=g[(g.model=='cmanp')&(g.adapt_steps==0)]
        for baseline in ['wknn','rbf']:
            b=g[(g.model==baseline)&(g.adapt_steps==0)][['domain','support_seed','mde_m']]
            j=a.merge(b,on=['domain','support_seed'],suffixes=('_main','_base'),validate='many_to_one')
            if j.empty:continue
            j['delta']=j.mde_m_main-j.mde_m_base
            v=j.groupby(['model_seed','support_seed']).delta.mean().unstack().to_numpy()
            boot=[]
            for _ in range(repeats):
                ix=rng.integers(len(v),size=len(v));jx=rng.integers(v.shape[1],size=v.shape[1])
                boot.append(float(v[ix][:,jx].mean()))
            rows.append({'stage':stage,'K':k,'comparison':f'cmanp-minus-{baseline}','delta_m':float(v.mean()),
                'lower95':float(np.quantile(boot,.025)),'upper95':float(np.quantile(boot,.975)),
                'scope':'fixed-floor conditional comparison; pilot is not confirmatory evidence'})
    return pd.DataFrame(rows)
