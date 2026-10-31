"""fidelity_linked_eval.py with the role-embedding log1p on co-occurrence counts off, so
tuple frequency reaches the embeddings (a local ablation; the engine is unchanged).
usage: python -I fidelity_linked_weights.py <none|nolog|noscale|both> <eval args...>
(noscale changes nothing with role embeddings: their scaler is already an identity.)
"""
import sys, runpy
import numpy as _np
mode = sys.argv[1]
engine = sys.argv[4]
sys.path.insert(0, engine)
import fidelityai.role_embedding as re_mod, fidelityai.features as ft
if mode in ("nolog", "both"):
    class P:
        def __getattr__(self, a): return getattr(_np, a)
        @staticmethod
        def log1p(x): return x
    re_mod.np = P()
if mode in ("noscale", "both"):
    class Identity:
        def __init__(self, **kw): pass
        def fit(self, X, y=None):
            self.scale_ = _np.ones(X.shape[1]); self.min_ = _np.zeros(X.shape[1]); return self
        def transform(self, X): return _np.asarray(X)
        def fit_transform(self, X, y=None): return self.fit(X).transform(X)
    ft.MinMaxScaler = Identity
sys.argv = ["eval"] + sys.argv[2:]
runpy.run_path("scripts/fidelity_linked_eval.py", run_name="__main__")
