from __future__ import annotations
import io, math, zipfile, urllib.request
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, dump, digest, stable_seed, file_hash

UCI_URL = 'https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip'
WAPS = [f'WAP{i:03d}' for i in range(1, 521)]
XY = ['LONGITUDE', 'LATITUDE']
TEST = ['B0F3', 'B1F3', 'B2F4']
DEV = ['B0F2', 'B1F2', 'B2F3']
FIT = ['B0F0', 'B0F1', 'B1F0', 'B1F1', 'B2F0', 'B2F1', 'B2F2']

def load_data(raw=ROOT / 'data/raw/ujiindoorloc.zip'):
    raw = Path(raw)
    reference = __import__('json').loads((ROOT/'data/DATA_SOURCE.json').read_text())
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        tmp = raw.with_suffix('.tmp')
        with urllib.request.urlopen(UCI_URL, timeout=120) as r, open(tmp, 'wb') as f:
            while chunk := r.read(2**20):
                f.write(chunk)
        if file_hash(tmp) != reference['zip_sha256']:
            tmp.unlink()
            raise ValueError('Downloaded UCI ZIP differs from pinned data. Audit a new version explicitly.')
        tmp.replace(raw)
    if file_hash(raw) != reference['zip_sha256']:
        raise ValueError('UCI ZIP checksum mismatch.')
    frames = []
    with zipfile.ZipFile(raw) as z:
        for split, filename, prefix in [('train', 'trainingData.csv', 'tr'), ('external', 'validationData.csv', 'va')]:
            content = z.read('UJIndoorLoc/' + filename)
            import hashlib
            if hashlib.sha256(content).hexdigest() != reference['csv_sha256'][filename]:
                raise ValueError('CSV hash mismatch')
            d = pd.read_csv(io.BytesIO(content))
            if d.shape[1] != 529 or list(d.columns[:520]) != WAPS:
                raise ValueError('Unexpected UJI schema')
            # Integer row IDs are stable: train 0..19936; validation starts at 19937.
            d['rid'] = np.arange(len(d)) + sum(len(x) for x in frames)
            d['split'] = split
            d['domain'] = 'B'+d.BUILDINGID.astype(str)+'F'+d.FLOOR.astype(str)
            # Use exact parsed float64 coordinates, never rounding before grouping.
            d['gid'] = [f'{dom}|{float(x).hex()}|{float(y).hex()}'
                        for dom,x,y in zip(d.domain,d.LONGITUDE,d.LATITUDE)]
            d['day'] = pd.to_datetime(d.TIMESTAMP, unit='s', utc=True).dt.strftime('%Y-%m-%d')
            d['cohort'] = [f'{p}|{u}|{t}' for p,u,t in zip(d.PHONEID,d.USERID,d.day)]
            if not np.isfinite(d[XY].to_numpy()).all():
                raise ValueError('Nonfinite coordinates')
            r = d[WAPS].to_numpy()
            if not (((r >= -104) & (r <= 0)) | (r == 100)).all():
                raise ValueError('RSSI outside declared range')
            frames.append(d)
    return pd.concat(frames, ignore_index=True)

@dataclass
class Transform:
    """Only fit source coordinates. No target normalization or feature statistics."""
    buildings: dict
    fit_row_hash: str

    @classmethod
    def fit(cls, d):
        if len(d) == 0 or not (d.split == 'train').all():
            raise ValueError('Coordinate scaler needs nonempty source training rows')
        buildings = {}
        # Position weighting prevents more repeated scans changing the coordinate origin.
        for b, g in d.drop_duplicates('gid').groupby('BUILDINGID'):
            y = g[XY].to_numpy(dtype=np.float64)
            center = y.mean(axis=0)
            scale = float(np.sqrt(((y-center)**2).mean()))
            if scale < 1e-8:
                raise ValueError('Degenerate source geometry')
            buildings[str(int(b))] = {'center': center.tolist(), 'scale': scale}
        return cls(buildings, digest(sorted(d.rid.astype(int).tolist())))

    def x(self, d):
        r = d[WAPS].to_numpy(dtype=np.float32)
        return np.where(r == 100, 0., (r+110.)/110.).astype(np.float32)

    def y(self, d):
        y = d[XY].to_numpy(dtype=np.float64).copy()
        for b in d.BUILDINGID.unique():
            spec = self.buildings[str(int(b))]; m = d.BUILDINGID.to_numpy() == b
            y[m] = (y[m] - np.asarray(spec['center'], dtype=np.float64))/spec['scale']
        return y.astype(np.float32)

    def inverse(self, y, building):
        spec = self.buildings[str(int(building))]
        return np.asarray(y, dtype=np.float64)*spec['scale']+np.asarray(spec['center'],dtype=np.float64)

    def as_dict(self):
        return {'buildings': self.buildings, 'fit_row_hash': self.fit_row_hash}


def make_domain_manifest(g, seed, query_fraction, support_seeds, max_k):
    ids = sorted(g.gid.unique())
    rng = np.random.default_rng(stable_seed(seed, str(g.domain.iloc[0]), 'query'))
    ids = list(np.asarray(ids)[rng.permutation(len(ids))])
    nq = math.ceil(query_fraction*len(ids))
    query_groups = set(ids[:nq]); pool = sorted(ids[nq:])
    if len(pool) < max_k:
        raise ValueError(f'{g.domain.iloc[0]} only has {len(pool)} support positions for K={max_k}')
    groups = {key: sorted(v.rid.astype(int).tolist()) for key,v in g.groupby('gid')}
    episodes = {}
    for es in support_seeds:
        erng = np.random.default_rng(stable_seed(seed,g.domain.iloc[0],int(es),'support'))
        order = erng.permutation(pool).tolist()
        # Select exactly one raw scan, not an uncounted average of repeats.
        episodes[str(es)] = [int(erng.choice(groups[key])) for key in order[:max_k]]
    query = sorted(g.loc[g.gid.isin(query_groups), 'rid'].astype(int).tolist())
    return {'query_rows': query, 'query_groups': sorted(query_groups), 'support_pool_groups': pool,
            'episodes': episodes, 'positions': len(ids), 'query_positions': nq,
            'support_pool_positions':len(pool)}


def make_protocol(d, cfg, matched=False):
    train = d[d.split == 'train']
    selected = train
    cohorts = {}
    if matched:
        parts = []
        for target in TEST:
            b = int(target[1]); s = train[(train.BUILDINGID == b)&(~train.domain.isin(TEST))]
            t = train[train.domain == target]
            common = sorted(set(s.cohort)&set(t.cohort))
            cohorts[str(b)] = common
            parts.append(train[(train.BUILDINGID == b)&train.cohort.isin(common)])
        selected = pd.concat(parts).sort_values('rid')
    domains = {}
    budgets = cfg['protocol']['matched_budgets'] if matched else cfg['protocol']['budgets']
    source = sorted(set(selected.domain)-set(TEST))
    for domain, g in selected.groupby('domain'):
        domains[domain] = make_domain_manifest(g, cfg['protocol']['split_seed'],
            cfg['protocol']['query_fraction'], cfg['protocol']['support_seeds'], max(budgets))
        domains[domain]['all_rows'] = sorted(g.rid.astype(int).tolist())
    manifest = {'version':'1.0', 'mode':'matched' if matched else 'primary',
        'data_zip_sha256': file_hash(ROOT/'data/raw/ujiindoorloc.zip'),
        'fit_domains':FIT if not matched else source, 'dev_domains':DEV if not matched else [],
        'test_domains':TEST, 'source_domains':source, 'cohorts':cohorts,
        'budgets':budgets, 'split_seed':cfg['protocol']['split_seed'],
        'support_seeds':cfg['protocol']['support_seeds'], 'domains':domains}
    manifest['hash'] = digest(manifest)
    validate_protocol(d, manifest)
    return manifest


def validate_protocol(d, m):
    body = {k:v for k,v in m.items() if k != 'hash'}
    if digest(body) != m['hash']:
        raise ValueError('Manifest changed since it was sealed')
    if set(m['source_domains']) & set(m['test_domains']):
        raise ValueError('Source-target floor leakage')
    for dom, spec in m['domains'].items():
        q = d.loc[spec['query_rows']]
        if not (q.domain == dom).all() or not (q.split == 'train').all():
            raise ValueError('Wrong query floor/split')
        for order in spec['episodes'].values():
            s = d.loc[order]
            if not (s.domain == dom).all() or not (s.split == 'train').all():
                raise ValueError('Wrong support floor/split')
            if s.gid.nunique() != len(s) or set(s.gid)&set(q.gid):
                raise ValueError('Repeated position or support-query leakage')
            if set(order)&set(spec['query_rows']):
                raise ValueError('Row leakage')


def domain_rows(d, m, domains):
    ids = [r for domain in domains for r in m['domains'][domain]['all_rows']]
    return d.loc[ids]


def audit(d, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for (split,dom),g in d.groupby(['split','domain']):
        xy=g.drop_duplicates('gid')[XY]
        rows.append({'split':split,'domain':dom,'rows':len(g),'positions':len(xy),
          'phone_ids':','.join(map(str,sorted(g.PHONEID.unique()))),
          'user_ids':','.join(map(str,sorted(g.USERID.unique()))),
          'first_utc':str(pd.to_datetime(g.TIMESTAMP.min(),unit='s',utc=True)),
          'last_utc':str(pd.to_datetime(g.TIMESTAMP.max(),unit='s',utc=True)),
          'x_span':float(xy.LONGITUDE.max()-xy.LONGITUDE.min()),
          'y_span':float(xy.LATITUDE.max()-xy.LATITUDE.min()),
          'visible_ap_count':int((g[WAPS].to_numpy()!=100).any(axis=0).sum())})
    pd.DataFrame(rows).to_csv(out/'dataset_audit.csv',index=False)
    cohort=d.groupby(['split','domain','PHONEID','USERID','day']).agg(
        rows=('rid','size'),positions=('gid','nunique')).reset_index()
    cohort.to_csv(out/'cohort_audit.csv',index=False)
    all_source=d[(d.split=='train')&(~d.domain.isin(TEST))]
    seen=(all_source[WAPS].to_numpy()!=100).any(0)
    novelty=[]
    for (split,dom),g in d[d.domain.isin(TEST)].groupby(['split','domain']):
        mask=g[WAPS].to_numpy()!=100
        novelty.append({'split':split,'domain':dom,'aps':int(mask.any(0).sum()),
            'aps_never_seen_in_source':int((mask.any(0)&~seen).sum()),
            'scan_fraction_with_source_unseen_ap':float(mask[:,~seen].any(1).mean())})
    dump(out/'ap_novelty.json',novelty)
    dump(out/'data_audit_summary.json',{'training_rows':int((d.split=='train').sum()),
        'external_rows':int((d.split=='external').sum()),
        'training_positions':int(d[d.split=='train'].gid.nunique()),
        'query_grouping':'exact float64 coordinates within building-floor; not a spatial independence guarantee',
        'external_semantics':'months-later mixed-device stress test, not pure cross-floor shift',
        'coordinate_unit':'UJI planar coordinate distance, conventionally reported as m; exact CRS unverified'})
