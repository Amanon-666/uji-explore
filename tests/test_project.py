from __future__ import annotations
import copy,json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from uji.common import ROOT,digest,seed_all,verify_upstream
from uji.data import load_data,make_protocol,validate_protocol,Transform,domain_rows,WAPS,TEST,XY
from uji.models import make_model,wknn,rbf_regression
from uji.metrics import metrics
from uji.run import config
from uji.engine import EpisodeSampler,predict_cmanp

@pytest.fixture(scope='module')
def d():return load_data()
@pytest.fixture(scope='module')
def cfg():return config(ROOT/'configs/uji.yaml','smoke')
@pytest.fixture(scope='module')
def m(d,cfg):return make_protocol(d,cfg)
@pytest.fixture(scope='module')
def tf(d,m):return Transform.fit(domain_rows(d,m,m['fit_domains']))
@pytest.fixture(scope='module')
def model(cfg):
    seed_all(7);return make_model('cmanp',cfg).eval()
@pytest.fixture(scope='module')
def tensors():
    g=torch.Generator().manual_seed(19)
    return torch.rand(1,10,520,generator=g),torch.randn(1,10,2,generator=g),torch.rand(1,7,520,generator=g)


def test_hashes_and_shape(d):
    assert len(d)==21048
    assert (d.split=='train').sum()==19937
    assert (d.split=='external').sum()==1111

def test_position_and_floor_counts(d):
    assert d[d.split=='train'].domain.nunique()==13
    assert d[(d.split=='train')&(d.domain=='B1F3')].gid.nunique()==50

def test_source_dev_target_disjoint(m):
    assert len(m['fit_domains'])==7 and len(m['dev_domains'])==3
    assert len(m['source_domains'])==10
    assert not set(m['fit_domains'])&set(m['dev_domains'])
    assert not set(m['source_domains'])&set(TEST)

def test_manifest_deterministic(d,cfg,m):assert make_protocol(d,cfg)['hash']==m['hash']

def test_nested_support_and_position_disjoint(d,m):
    validate_protocol(d,m)
    for spec in m['domains'].values():
        for order in spec['episodes'].values():
            assert len(order)==20
            assert set(order[:5])<=set(order[:10])<=set(order[:20])
            assert not set(d.loc[order].gid)&set(d.loc[spec['query_rows']].gid)

def test_manifest_tampering_detected(d,m):
    bad=copy.deepcopy(m);bad['domains']['B0F3']['query_rows'][0]=0
    with pytest.raises(ValueError):validate_protocol(d,bad)

def test_exact_duplicates_not_split(d,m):
    for dom,spec in m['domains'].items():
        qgroups=set(d.loc[spec['query_rows']].gid)
        allrows=d[(d.domain==dom)&(d.split=='train')]
        assert set(allrows[allrows.gid.isin(qgroups)].rid)==set(spec['query_rows'])

def test_query_fraction(m):
    import math
    for spec in m['domains'].values():assert spec['query_positions']==math.ceil(.3*spec['positions'])

def test_scaler_source_provenance(d,m,tf):
    ids=domain_rows(d,m,m['fit_domains']).rid.astype(int).tolist()
    assert tf.fit_row_hash==digest(sorted(ids))

def test_scaler_target_labels_not_used(d,m,tf):
    poison=d.copy();poison.loc[poison.domain.isin(TEST),XY]=1e10
    other=Transform.fit(domain_rows(poison,m,m['fit_domains']))
    assert other.as_dict()==tf.as_dict()

def test_scaler_roundtrip_float64(d,tf):
    g=d[(d.domain=='B0F0')].iloc[:10]
    np.testing.assert_allclose(tf.inverse(tf.y(g),0),g[XY],atol=1e-4,rtol=0)

def test_no_metadata_features(d,tf):
    a=d.iloc[:10].copy();b=a.copy()
    for name in ['PHONEID','TIMESTAMP','FLOOR','USERID','SPACEID','RELATIVEPOSITION']:
        b[name]=12345
    np.testing.assert_array_equal(tf.x(a),tf.x(b))

def test_rss_missing_encoding(d,tf):
    a=d.iloc[:1].copy();a[WAPS]=100;a[WAPS[0]]=-104;a[WAPS[1]]=0
    x=tf.x(a)
    assert x.shape==(1,520)
    assert x[0,0]>0 and x[0,1]==1 and (x[0,2:]==0).all()

def test_sampler_shapes(d,m,tf,cfg):
    sampler=EpisodeSampler(domain_rows(d,m,m['fit_domains']),tf,0)
    xc,yc,xq,yq=sampler.sample(2,20,16,'cpu')
    assert xc.shape==(2,20,520) and yq.shape==(2,16,2)

def test_support_permutation(model,tensors):
    xc,yc,xq=tensors;p=torch.randperm(10)
    with torch.no_grad():
        a=model.predict(xc,yc,xq);b=model.predict(xc[:,p],yc[:,p],xq)
    torch.testing.assert_close(a.mean,b.mean,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(a.stddev,b.stddev,rtol=2e-5,atol=2e-6)

def test_incremental_matches_full_conditioning(model,tensors):
    xc,yc,xq=tensors
    with torch.no_grad():
        _,state=model.get_context_encoding(SimpleNamespace(xc=xc[:,:5],yc=yc[:,:5]),return_state=True)
        enc,_=model.update(xc[:,5:],yc[:,5:],state)
        a=model.predict(xc,yc,xq);b=model.predict(xc,yc,xq,context_encodings=enc)
    torch.testing.assert_close(a.mean,b.mean,rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(a.stddev,b.stddev,rtol=3e-5,atol=3e-6)

def test_query_chunk_equivalence(model,tensors):
    a,_=predict_cmanp(model,*tensors,128);b,_=predict_cmanp(model,*tensors,2)
    np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)

def test_query_order_equivariance(model,tensors):
    xc,yc,xq=tensors;ix=torch.randperm(7)
    with torch.no_grad():a=model.predict(xc,yc,xq).mean;b=model.predict(xc,yc,xq[:,ix]).mean
    torch.testing.assert_close(a[:,ix],b,rtol=2e-5,atol=2e-6)

def test_query_labels_unnecessary(model,tensors):
    # No query-label argument exists anywhere in the prediction API.
    with torch.no_grad():
        p=model.predict(*tensors)
    assert p.mean.shape==(1,7,2) and torch.isfinite(p.mean).all()

def test_backprop_finite(cfg,tensors):
    model=make_model('cmanp',cfg);xc,yc,xq=tensors
    loss=-model.predict(xc,yc,xq).log_prob(torch.zeros(1,7,2)).sum(-1).mean();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

def test_wknn_known_points():
    x=np.array([[0.,0.],[1.,1.]]);y=np.array([[2.,3.],[9.,8.]])
    np.testing.assert_allclose(wknn(x,y,x,3),y)

def test_wknn_exact_ties():
    x=np.zeros((2,3));y=np.array([[2.,3.],[4.,5.]])
    np.testing.assert_allclose(wknn(x,y,np.zeros((1,3)),2),[[3.,4.]])

def test_kernel_duplicates_stable():
    result=rbf_regression(np.zeros((5,3)),np.arange(10).reshape(5,2),np.ones((7,3)),1.,.01)
    assert result.shape==(7,2) and np.isfinite(result).all()

def test_position_weighted_metric():
    y=np.zeros((10,2));pred=np.zeros((10,2));pred[:9,0]=1;pred[9,0]=9
    result,_=metrics(y,pred,['a']*9+['b'])
    assert result['mde_m']==pytest.approx(5.)
    assert result['sample_mde_m']==pytest.approx(1.8)

def test_gaussian_calibration_units():
    y=np.zeros((2,2));r,_=metrics(y,y,['a','b'],std=np.ones((2,2)),scale=2.)
    assert r['nll_planar']==pytest.approx(np.log(2*np.pi)+2*np.log(2))
    assert r['ellipse95_coverage']==1

def test_matched_cohort_feasibility(d,cfg):
    matched=make_protocol(d,cfg,True)
    assert matched['budgets']==[5]
    spec=matched['domains']['B1F3']
    assert spec['positions']==14 and spec['support_pool_positions']==9
    for b,cohorts in matched['cohorts'].items():
        source=domain_rows(d,matched,[s for s in matched['source_domains'] if s[1]==b])
        target=domain_rows(d,matched,[s for s in TEST if s[1]==b])
        assert set(source.cohort)==set(target.cohort)==set(cohorts)

def test_official_code_unmodified():assert verify_upstream()=='8961cd940153d76918f401acc60aa858101a8949'


def test_episode_sampler_resume(d,m,tf):
    a=EpisodeSampler(domain_rows(d,m,m['fit_domains']),tf,41)
    a.sample(2,5,16,'cpu');state=copy.deepcopy(a.rng.bit_generator.state)
    expected=a.sample(2,10,16,'cpu')
    b=EpisodeSampler(domain_rows(d,m,m['fit_domains']),tf,99)
    b.rng.bit_generator.state=state
    actual=b.sample(2,10,16,'cpu')
    for x,y in zip(expected,actual):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_source_training_rejects_target(d,m,cfg,tmp_path):
    from uji.engine import train
    with pytest.raises(ValueError,match='Target domain'):
        train(d,m,cfg,'mlp',0,tmp_path,'cpu',m['test_domains'],steps=1)
