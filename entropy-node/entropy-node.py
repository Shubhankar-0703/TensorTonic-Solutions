import numpy as np

def entropy_node(y):
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    _,counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    log_probs = np.log2(probs, where=probs > 0, out=np.zeros_like(probs, dtype=float))
    return float(-np.sum(probs * log_probs))