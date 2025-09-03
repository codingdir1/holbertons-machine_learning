#!/usr/bin/env python3

def update_variables_momentum(alpha, beta1, var, grad, v):
    Vdv = beta1 * v + (1 - beta1) * grad
    var = var - alpha * Vdv
    return var, Vdv
