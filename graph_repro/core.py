"""GConvLoc reconstruction using official PyG GAT parameters and attention.

Paper: doi:10.1587/transinf.2022EDL8081. This is NOT recovered author code.
The low-memory aggregation is mathematically identical to PyG GATConv with
one head, dropout=0 and explicit self loops; parity is tested, including gradients.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax
from sklearn.neighbors import NearestNeighbors

WAPS = [f'WAP{i:03d}' for i in range(1, 521)]
XY = ['LONGITUDE', 'LATITUDE']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csvs(directory):
    """Validate exact official CSV bytes; no silent substitution."""
    expected = {
        'trainingData.csv': '45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168',
        'validationData.csv': '5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0',
    }
    out = []
    for split, name in enumerate(expected):
        path = Path(directory) / name
        if digest(path) != expected[name]:
            raise ValueError(f'Official CSV checksum mismatch: {path}')
        df = pd.read_csv(path)
        df['rid'] = np.arange(len(df)) + (0 if split == 0 else 19937)
        df['official_split'] = 'train' if split == 0 else 'validation'
        out.append(df)
    return out


def source_split(df, seed=0, fraction=0.1):
    """Within-location stratified scan holdout; documented reconstruction choice.

    NOT a claim of recovering CNNLoc's precise uniform-sampling implementation.
    Only source training rows may be passed. Shared locations are intentional in
    this conventional validation; spatial extrapolation is a separate protocol.
    """
    if not (df.official_split == 'train').all():
        raise ValueError('Official validation must never enter development.')
    rng = np.random.default_rng(seed)
    fit, dev = [], []
    for _, g in df.groupby(['BUILDINGID', 'FLOOR', *XY], sort=True):
        ids = rng.permutation(g.index.to_numpy())
        n = min(len(ids)-1, max(1, int(round(len(ids)*fraction))))
        dev.extend(ids[:n]); fit.extend(ids[n:])
    return df.loc[sorted(fit)].copy(), df.loc[sorted(dev)].copy()


class Preprocess:
    """Statistics fit on source-fit rows only. Raw coordinates stay float64."""
    def __init__(self, mode='gconvloc'):
        self.mode = mode

    def fit(self, df):
        if not (df.official_split == 'train').all():
            raise ValueError('Preprocessing is source-training-only.')
        x = df[WAPS].to_numpy(dtype=np.float64)
        if self.mode == 'gconvloc':
            x[x == 100] = -104
            self.xloc = np.full(520, x.min())
            self.xscale = np.full(520, max(x.max()-x.min(), 1.0))
        elif self.mode == 'jprl':
            self.xloc, self.xscale = x.mean(0), x.std(0)
            self.xscale[self.xscale < 1e-12] = 1.0
        else:
            raise ValueError(self.mode)
        y = df[XY].to_numpy(dtype=np.float64)
        self.yloc = y.min(0)
        self.yscale = np.maximum(y.max(0)-self.yloc, 1.0)
        self.fit_ids = df.rid.astype(int).tolist()
        return self

    def x(self, df):
        x = df[WAPS].to_numpy(dtype=np.float64)
        missing = x == 100
        if self.mode == 'gconvloc':
            x[missing] = -104
        x = (x-self.xloc)/self.xscale
        if self.mode == 'gconvloc':
            x[missing] = 0.0
        return x.astype(np.float32)

    def y(self, df):
        return ((df[XY].to_numpy(dtype=np.float64)-self.yloc)/self.yscale).astype(np.float32)

    def inverse(self, y):
        return np.asarray(y, dtype=np.float64)*self.yscale+self.yloc

    def state(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}

    @classmethod
    def restore(cls, state):
        p = cls(state['mode'])
        for k, v in state.items():
            setattr(p, k, np.asarray(v) if k in ['xloc','xscale','yloc','yscale'] else v)
        return p


def graph_edges(reference, query=None, k=23):
    """Edges point reference -> destination, NEVER query -> reference/query.

    Exclude reference self from kNN then add exactly one self loop to every node.
    Cosine distance is evaluated on nonnegative normalized fingerprints.
    """
    n = len(reference)
    if n < 2:
        raise ValueError('At least two reference fingerprints required.')
    k = min(k, n-1)
    nnbr = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=1)
    nnbr.fit(reference)
    # Chunk queries to avoid materializing an N x N distance matrix.
    src, dst = [], []
    for start in range(0, n, 512):
        idx = nnbr.kneighbors(reference[start:start+512], return_distance=False)
        for j, neighbors in enumerate(idx):
            own = start+j
            take = neighbors[neighbors != own][:k]
            src.extend(take.tolist()); dst.extend([own]*len(take))
    total = n
    if query is not None and len(query):
        for start in range(0, len(query), 512):
            idx = nnbr.kneighbors(query[start:start+512], n_neighbors=k, return_distance=False)
            src.extend(idx.reshape(-1).tolist())
            dst.extend(np.repeat(np.arange(start, start+len(idx))+n, k).tolist())
        total += len(query)
    src.extend(range(total)); dst.extend(range(total))
    return torch.tensor([src, dst], dtype=torch.long)


class _Aggregate(torch.autograd.Function):
    """Exact edge-weighted sum with bounded E*D temporary memory."""
    @staticmethod
    def forward(ctx, x, weight, edge):
        src, dst = edge
        adj = torch.sparse_coo_tensor(torch.stack([dst, src]), weight, (len(x), len(x)))
        out = torch.sparse.mm(adj, x)
        ctx.save_for_backward(x, weight, edge)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, edge = ctx.saved_tensors
        src, dst = edge
        reverse = torch.sparse_coo_tensor(torch.stack([src, dst]), weight, (len(x), len(x)))
        gx = torch.sparse.mm(reverse, grad)
        gw = torch.empty_like(weight)
        for a in range(0, len(weight), 4096):
            b = a+4096
            gw[a:b] = (grad[dst[a:b]]*x[src[a:b]]).sum(-1)
        return gx, gw, None


class MemoryGAT(GATConv):
    """PyG GATConv-compatible, 1-head homogeneous special case, zero dropout."""
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, heads=1, dropout=0,
                         add_self_loops=False)

    def forward(self, x, edge_index):
        h = self.lin(x).view(-1, self.heads, self.out_channels)
        a_s = (h*self.att_src).sum(-1).squeeze(-1)
        a_t = (h*self.att_dst).sum(-1).squeeze(-1)
        src, dst = edge_index
        logits = torch.nn.functional.leaky_relu(a_s[src]+a_t[dst], self.negative_slope)
        alpha = softmax(logits, dst, num_nodes=len(x))
        out = _Aggregate.apply(h.squeeze(1), alpha, edge_index)
        return out if self.bias is None else out+self.bias


class GConvLoc(nn.Module):
    """Paper Table 2 widths; activation/output choices are explicitly audited."""
    def __init__(self, activation='relu', output='linear', backend='memory'):
        super().__init__()
        make = MemoryGAT if backend == 'memory' else lambda a,b: GATConv(a,b,heads=1,dropout=0,add_self_loops=False)
        self.gat1, self.gat2 = make(520,256), make(256,128)
        self.fc1, self.fc2 = nn.Linear(128,64), nn.Linear(64,2)
        self.act = nn.ReLU() if activation == 'relu' else nn.ELU()
        self.output = nn.Sigmoid() if output == 'sigmoid' else nn.Identity()

    def forward(self, x, edge):
        h = self.act(self.gat1(x,edge))
        h = self.act(self.gat2(h,edge))
        return self.output(self.fc2(self.act(self.fc1(h))))


class JPRLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(520,260), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(260,2), nn.Sigmoid())

    def forward(self,x):
        h = self.encoder(x)
        return self.head(h), h


def jprl_l2(h,y,domain,epsilon=1e-3):
    """JPRL regression Eqs (6)-(10), NOT RCS, MMD, HSIC, or an adversarial loss.

    All batch samples are kernel centers. Float64 solve stabilizes nearly equal
    samples. epsilon is the paper's unspecified numerical ridge (choice audited).
    """
    h, y = h.double(), y.double()
    d = (h[:,None]-h[None,:]).square().sum(-1)
    d = d + (y[:,None]-y[None,:]).square().sum(-1)
    kernel = torch.exp(-torch.pi*d/2)
    same = (domain[:,None] == domain[None,:]).to(h.dtype)
    p = kernel*same
    H = torch.exp(-torch.pi*d/4)*same
    b = p.mean(1)-kernel.mean(1)*same.mean(1)
    theta = torch.linalg.solve(H+epsilon*torch.eye(len(h),device=h.device,dtype=h.dtype),b)
    return b@theta-0.5*theta@H@theta


def metrics(pred,truth,scale):
    pred, truth = np.asarray(pred,dtype=np.float64), np.asarray(truth,dtype=np.float64)
    d = pred-truth
    e = np.linalg.norm(d,axis=1)
    return dict(n=len(e),mde_m=float(e.mean()),std_error_m=float(e.std()),
                median_m=float(np.median(e)),p90_m=float(np.quantile(e,0.9)),
                coordinate_mae_m=float(np.abs(d).mean()),coordinate_mse_m2=float((d*d).mean()),
                coordinate_rmse_m=float(np.sqrt((d*d).mean())),
                normalized_sum_mae=float(np.abs(d/scale).mean(0).sum()))


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
