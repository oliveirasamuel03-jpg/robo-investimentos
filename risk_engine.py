import numpy as np

def position_size(volatility, max_risk=0.02):
    return max_risk / (volatility + 1e-9)

def apply_risk_limits(weights):
    weights = np.clip(weights, -1, 1)
    return weights / (np.sum(np.abs(weights)) + 1e-9)