import numpy

def correlation(C):
    if (type(C) != numpy.ndarray):
        raise ValueError("C must be a numpy.ndarray")
    elif (len(C.shape) != 2 or C.shape[0] != C.shape[1]):
        raise ValueError("C must be a 2D square matrix")

    corr_matrix = numpy.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            corr_matrix[i, j] = C[i, j] / ((C[i, i] ** 0.5) * (C[j, j] ** 0.5))
    return corr_matrix
