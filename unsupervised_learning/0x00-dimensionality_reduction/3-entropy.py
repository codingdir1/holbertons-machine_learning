#!/urs/bin/env python3

import numpy as np

def HP(Di, beta):

    Pi = np.exp(-1 * Di * beta) / np.sum(np.exp(-1 * Di * beta))

    Hi = -1 * np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
