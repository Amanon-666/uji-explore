from __future__ import annotations
import hashlib, json, random, sys, platform, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')

def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def seed_all(seed, threads=2):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def device_name(request):
    if request == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for smoke/pilot.')
    return request

def environment():
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda, 'threads': torch.get_num_threads()}

def upstream_imports():
    """Use the pinned upstream modules unchanged; do not install this shim globally."""
    for p in [ROOT / 'third_party/constant-memory-anp/regression', ROOT / 'compat']:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

def verify_upstream():
    manifest = json.loads((ROOT / 'third_party/manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if file_hash(ROOT / 'third_party/constant-memory-anp' / name) != expected:
            raise ValueError(f'Upstream source modified: {name}')
    return manifest['commit']


def source_fingerprint():
    files = sorted((ROOT/'uji').glob('*.py')) + sorted((ROOT/'compat').glob('*.py'))
    return digest({str(p.relative_to(ROOT)):file_hash(p) for p in files})
