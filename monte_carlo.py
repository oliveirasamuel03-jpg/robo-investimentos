import numpy as np

def monte_carlo(returns, n=1000):
    sims = []

    for _ in range(n):
        sample = np.random.choice(returns, len(returns))
        sims.append(np.sum(sample))

    return np.array(sims)