#!/urs/bin/env python3

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost

def tsne(X, ndims = 2, idims = 50, perplexity = 30.0, iterations = 1000, lr = 500):
    Yt_1, Yt_2 = None, None
    
    X = pca(X, idims)
    P = P_affinities(X = X, perplexity = perplexity)
    Y = np.random.normal(loc = 0.0, scale = 1e-2, size = (X.shape[0], ndims))

    for i in range(iterations + 1):
        factor = 1
        if i < 100:
            factor = 4

        dY, Q = grads(Y, factor * P)

        if i == 0:
            Yt_1, Yt_2 = np.zeros(Y.shape), np.zeros(Y.shape)
        if i == 1:
            Yt_1 = Y
        else:
            Yt_2 = Yt_1
            Yt_1 = Y

        if i >= 20:
            a_t = 0.8
        else:
            a_t = 0.5

        Y = Yt_1 - (lr * dY) + (a_t * (Yt_1 - Yt_2))
        Y = Y - np.mean(Y, axis = 0)

        if (i + 1) % 100 == 0:
        c = cost(factor * P, Q)
        print(f"Cost at iteration {i + 1}: {c}")
    return Y
