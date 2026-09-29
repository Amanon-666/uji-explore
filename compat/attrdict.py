"""Minimal attribute dictionary for legacy upstream model/data imports.

Only implements the operations used by the vendored CMANP and GP sampler.
This is a compatibility shim, not a replacement for arbitrary attrdict programs.
"""
class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
