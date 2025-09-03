#!/usr/bin/env python3

import numpy as np

def create_confusion_matrix(labels, logits):
    matrix = np.zeros(shape = (labels.shape[1], labels.shape[1]))
    for i in range(labels.shape[0]):
        truth = np.argmax(labels[i])
        pred = np.argmax(logits[i])
        matrix[truth][pred] += 1 
    return matrix
