#!/usr/bin/env python3

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP

def P_affinities(X, tol = 1e-5, perplexity = 30.0):

    D, P, betas, H = P_init(X, perplexity)

    n = X.shape[0]
    for i in range(n):
        # Remove p(i, i)
        mask = np.ones(n, dtype = bool)
        mask[i] = False
        Di = D[i]
        Di = Di[mask]
        
        # Tune the beta parameter
        high, low = None, None
        while True:
            Hi, Pi = HP(Di, betas[i])
            if Hi - H > tol:
                low = betas[i, 0]
                if high == None:
                    betas[i] *= 2
                else:
                    betas[i] = (betas[i] + high) / 2
            elif H - Hi > tol:
                high = betas[i, 0]
                if low == None:
                    betas[i] /= 2
                else:
                    betas[i] = (betas[i] + low) / 2
            else:
                Pi = np.insert(Pi, i, 0)
                P[i] += Pi
                break
    
    #Introduce sysmmetry
    P = (P + P.T) / (2 * n)
    return P
