import numpy as np

def risk_parity(vols):
    inv = 1 / (vols + 1e-9)
    return inv / inv.sum()