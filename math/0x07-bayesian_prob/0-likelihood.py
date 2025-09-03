import numpy
import math

def likelihood(x, n, P):
    if (n <= 0 or type(n) != int):
        raise ValueError("n must be a positive integer")
    elif (type(x) != int or x < 0):
        raise ValueError("x must be an integer that is greater than or equal to 0")
    elif (x > n):
        raise ValueError("x cannot be greater than n")
    elif (type(P) != numpy.ndarray or len(P.shape) != 1):
        raise ValueError ("P must be a 1D numpy.ndarray")

    likelihood_array = numpy.zeros(P.shape[0])
    for i in range(likelihood_array.shape[0]):
        if (P[i] < 0 or P[i] > 1):
            raise ValueError("All values in P must be in the range [0, 1]")
        likelihood_array[i] = (math.factorial(n) / (math.factorial(n - x) * math.factorial(x))) * (P[i] ** x) * ((1 - P[i]) ** (n - x))
    return likelihood_array
