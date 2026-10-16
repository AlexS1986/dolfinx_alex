#!/usr/bin/env python3
"""OpenBLAS positive control for the alex-dolfinx container (OpenBLAS 0.3.20, DYNAMIC_ARCH).
On the Lichtenberg i02 nodes (Sapphire Rapids) the auto-detected "Cooperlake" kernel returns wrong numbers;
with OPENBLAS_CORETYPE=SkylakeX (passed as APPTAINERENV_OPENBLAS_CORETYPE) the results are correct.
Judge by the dgemm/dgesv lines: expected ~1e-13 (ok), ~1e1 (dgemm) / ~1e4 (dgesv) when the kernel is wrong.
Ref: HRZ, OpenBLAS_SPR_Knoten_20260905.pdf"""
import ctypes
import ctypes.util
import os
import platform
import socket

import numpy as np

print("host:", socket.gethostname(), " cpu:", platform.processor() or "?")
try:
    model = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].split(":", 1)[1].strip()
    print("cpu model:", model)
except Exception:
    pass
print("OPENBLAS_CORETYPE env:", os.environ.get("OPENBLAS_CORETYPE", "<not set>"))

import glob

core = "?"
candidates = [ctypes.util.find_library(n) for n in ("openblas", "openblasp", "openblas64_")]
candidates += glob.glob(os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs", "libopenblas*"))   # pip wheels
candidates += glob.glob(os.path.join(os.path.dirname(np.__file__), "..", "scipy.libs", "libopenblas*"))
candidates += glob.glob("/usr/lib/x86_64-linux-gnu/libopenblas*.so*") + glob.glob("/usr/lib/x86_64-linux-gnu/openblas*/libopenblas*.so*")
for lib in [c for c in candidates if c]:
    try:
        lo = ctypes.CDLL(lib)
        lo.openblas_get_corename.restype = ctypes.c_char_p
        core = lo.openblas_get_corename().decode()
        print("openblas library:", lib)
        try:
            lo.openblas_get_config.restype = ctypes.c_char_p
            print("openblas config:", lo.openblas_get_config().decode())
        except Exception:
            pass
        break
    except Exception:
        pass
print("core:", core)

rng = np.random.default_rng(0)
n = 300
A = rng.standard_normal((n, n))
B = rng.standard_normal((n, n))
C_blas = A @ B                                                  # dgemm
C_ref = np.sum(A[:, :, None] * B[None, :, :], axis=1)          # element-wise, no BLAS
print(f"dgemm max deviation from reference: {np.max(np.abs(C_blas - C_ref)):.3e}   (expected ~1e-13)")

A = rng.standard_normal((n, n)) + n * np.eye(n)
b = rng.standard_normal(n)
x = np.linalg.solve(A, b)                                       # dgesv
r = np.sum(A * x[None, :], axis=1) - b                          # residual without BLAS
print(f"dgesv residual |Ax-b|_max: {np.max(np.abs(r)):.3e}   (expected ~1e-13)")
ok = np.max(np.abs(C_blas - C_ref)) < 1e-8 and np.max(np.abs(r)) < 1e-8
print("RESULT:", "OK" if ok else "WRONG RESULTS -> BLAS kernel broken on this node")
