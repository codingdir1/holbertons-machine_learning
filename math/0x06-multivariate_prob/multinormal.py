import numpy
import math

class MultiNormal():
    def __init__(self, data):
        if (not(type(data) == numpy.ndarray and len(data.shape) == 2)):
            raise ValueError("data must be a 2D numpy.ndarray")
        elif(data.shape[0] < 2):
            raise ValueError("data must contain multiple data points")

        self.mean = numpy.zeros((data.shape[0], 1))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.mean[i, 0] += data[i, j]
        self.mean /= data.shape[1]

        self.cov = numpy.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[1]):
            self.cov += numpy.outer(data[:, i] - self.mean[:, 0], data[:, i] - self.mean[:, 0])
        self.cov /= data.shape[1]

    def pdf(self, x):
        if (type(x) != numpy.ndarray):
            raise ValueError("x must be numpy.ndarray")
        elif (x.shape != self.mean.shape):
            raise ValueError("x must have the shape ({self.mean.shape[0]}, 1)")

        return (1 / (((2 * math.pi) ** (x.shape[0] / 2)) * (numpy.linalg.det(self.cov) ** 0.5))) * math.exp(-0.5 * numpy.dot(numpy.dot(x[:, 0] - self.mean[:, 0], numpy.linalg.inv(self.cov)), x[:, 0] - self.mean[:, 0]))
