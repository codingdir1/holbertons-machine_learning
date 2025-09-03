#!/usr/bin/env python3

def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    Vdv = beta1 * v + (1 - beta1) * grad
    Sdv = beta2 * s + (1 - beta2) * (grad ** 2)

    Vdv_corr = Vdv / (1 - beta1 ** t)
    Sdv_corr = Sdv / (1 - beta2 ** t)

    var = var - alpha * Vdv_corr / (Sdv_corr ** 0.5 + epsilon)
    return var, Vdv, Sdv
