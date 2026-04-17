import numpy as np

def generate_signals(probabilities, threshold=0.55):
    return (probabilities > threshold).astype(int)

def rank_assets(probabilities):
    return np.argsort(probabilities)[::-1]