import numpy

def mean_cov(X):
    if (type(X) != numpy.ndarray or len(X.shape) != 2):
        raise ValueError("X must be a 2D numpy.ndarray")
    elif (X.shape[0] < 2):
        raise ValueError("X must contain multiple data points")

    mean = numpy.zeros((1, X.shape[1]))
    for x_vector in X:
        for i in range(len(x_vector)):
            mean[0][i] += x_vector[i]
    mean /= X.shape[0]

    cov = numpy.zeros((X.shape[1], X.shape[1]))
    for x_vector in X:
        cov += numpy.outer((x_vector - mean), (x_vector - mean))
    cov /= X.shape[0]

    return mean, cov
