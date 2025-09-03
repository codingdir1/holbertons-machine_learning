#!/usr/bin/env python3

def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    Sdv = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / ((Sdv ** 0.5) + epsilon)
    return var, Sdv
