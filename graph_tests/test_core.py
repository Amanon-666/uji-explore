import numpy as np
import torch
from torch_geometric.nn import GATConv
from graph_repro.core import MemoryGAT, graph_edges, GConvLoc, jprl_l2, metrics

def test_pyg_forward_and_backward_equivalence():
    torch.manual_seed(2)
    x=torch.randn(19,7,dtype=torch.float64,requires_grad=True)
    edge=graph_edges(np.random.default_rng(1).random((19,7)).astype('float32'),k=3)
    fast=MemoryGAT(7,5).double()
    original=GATConv(7,5,heads=1,dropout=0,add_self_loops=False).double()
    original.load_state_dict(fast.state_dict())
    y=fast(x,edge); z=original(x,edge)
    torch.testing.assert_close(y,z,atol=1e-10,rtol=1e-9)
    y.square().sum().backward(); gx=x.grad.clone(); x.grad=None
    z.square().sum().backward()
    torch.testing.assert_close(x.grad,gx,atol=1e-9,rtol=1e-8)
    for a,b in zip(fast.parameters(),original.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-9,rtol=1e-8)

def test_graph_directions_and_self_loops():
    rng=np.random.default_rng(1);r=rng.random((10,520)).astype('float32');q=rng.random((3,520)).astype('float32')
    edge=graph_edges(r,q,3).numpy();s,t=edge
    assert np.all((s<10)|(s==t))
    assert np.sum(s==t)==13
    assert len(set(zip(s,t)))==len(s)

def test_query_batch_invariance_and_reference_isolation():
    torch.manual_seed(2);rng=np.random.default_rng(3)
    r=rng.random((9,520)).astype('float32');q=rng.random((4,520)).astype('float32')
    net=GConvLoc().eval()
    with torch.no_grad():
        all_y=net(torch.from_numpy(np.r_[r,q]),graph_edges(r,q,3))
        ref=net(torch.from_numpy(r),graph_edges(r,k=3))
        one=net(torch.from_numpy(np.r_[r,q[:1]]),graph_edges(r,q[:1],3))
    torch.testing.assert_close(ref,all_y[:9],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(one[-1],all_y[9],atol=1e-6,rtol=1e-5)

def test_jprl_single_domain_is_zero():
    h=torch.randn(5,4,requires_grad=True);y=torch.randn(5,2)
    loss=jprl_l2(h,y,torch.zeros(5,dtype=torch.long))
    assert abs(float(loss.detach()))<1e-12
    loss.backward();assert torch.isfinite(h.grad).all()

def test_jprl_gradient_and_permutation():
    torch.manual_seed(1)
    h=torch.randn(6,3,dtype=torch.float64,requires_grad=True)/3
    y=torch.randn(6,2,dtype=torch.float64);d=torch.tensor([0,0,1,1,2,2]);p=torch.randperm(6)
    torch.testing.assert_close(jprl_l2(h,y,d),jprl_l2(h[p],y[p],d[p]))
    assert torch.autograd.gradcheck(lambda h:jprl_l2(h,y,d),(h,))

def test_metric_units():
    m=metrics([[3,4]],[[0,0]],np.array([10,20]))
    assert m['mde_m']==5
    assert m['coordinate_mse_m2']==12.5
    assert m['normalized_sum_mae']==0.5


def make_frame(n=12,split='train'):
    import pandas as pd
    from graph_repro.core import WAPS
    x=np.full((n,520),100.0);x[:,0]=-70-np.arange(n)%3
    d=pd.DataFrame(x,columns=WAPS)
    d['LONGITUDE']=-7500+np.arange(n)//3;d['LATITUDE']=4864900.+np.arange(n)//3
    d['BUILDINGID']=0;d['FLOOR']=np.arange(n)%2;d['rid']=np.arange(n)
    d['official_split']=split
    return d

def test_preprocess_source_only_and_coordinate_precision():
    import pytest
    from graph_repro.core import Preprocess,XY
    d=make_frame();p=Preprocess().fit(d);before=p.state()
    q=d.copy();q.LATITUDE+=.123456
    np.testing.assert_allclose(p.inverse(p.y(q)),q[XY],atol=1e-6,rtol=0)
    p.x(q);assert p.state()==before
    np.testing.assert_equal(p.x(d)[:,1:],0.)
    with pytest.raises(ValueError):Preprocess().fit(make_frame(split='validation'))

def test_jprl_preserves_sentinel_for_stated_zscore_policy():
    from graph_repro.core import Preprocess
    d=make_frame();d.loc[0,'WAP001']=100
    p=Preprocess('jprl').fit(d)
    assert p.xloc[0]==d.WAP001.mean()

def test_domain_partitions():
    from graph_repro.run import partitions
    t=make_frame();e=make_frame(split='validation');e['rid']+=19937
    source,target,stress=partitions(t,e,'floor-dg',1)
    assert set(source.FLOOR)=={0};assert set(target.FLOOR)=={1}
    assert set(source.rid).isdisjoint(target.rid)
    assert set(stress.official_split)=={'validation'}

def test_source_split_complete_disjoint():
    from graph_repro.core import source_split
    d=make_frame(24);f,v=source_split(d)
    assert set(f.rid).isdisjoint(v.rid)
    assert set(f.rid)|set(v.rid)==set(d.rid)

def test_cnnloc_ids_manifest():
    import json
    from pathlib import Path
    ids=json.loads((Path(__file__).parents[1]/'data/raw/graph/cnnloc_split_audit.json').read_text())['dev_ids']
    assert len(ids)==len(set(ids))==2132
    assert min(ids)>=0 and max(ids)<19937
