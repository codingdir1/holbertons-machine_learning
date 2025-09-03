import numpy

likelihood = __import__('0-likelihood').likelihood

def intersection(x, n, P, Pr):
    if (n <= 0 or type(n) != int):
        raise ValueError("n must be a positive integer")
    elif (x < 0 or type(x) != int):
        raise ValueError("x must be an integer that is greater than or equal to 0")
    elif (x > n):
        raise ValueError("x cannot be greater than n")
    elif (type(P) != numpy.ndarray or len(P.shape) != 1):
        raise ValueError("P must be a 1D numpy.ndarray")
    elif (type(Pr) != numpy.ndarray or Pr.shape != P.shape):
        raise ValueError("Pr must be a numpy.ndarray with the same shape as P")
    sum_Pr = 0
    for i in range(P.shape[0]):
        if (P[i] < 0 or P[i] > 1):
            raise ValueError("All values in P must be in the range [0, 1]")
        elif (Pr[i] < 0 or Pr[i] > 1):
            raise ValueError("All values in Pr must be in the range [0, 1]")
        sum_Pr += Pr[i]
    if (numpy.isclose(sum_Pr, 1) == False):
        raise ValueError("Pr must sum to 1")

    likelihood_array = likelihood(x, n, P)    
    intersection_array = likelihood_array * Pr

    return intersection_array
