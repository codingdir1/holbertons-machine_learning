class Exponential:
    def __init__(self, data=None, lambtha=1.):
        if (data == None):
            if (lambtha > 0):
                self.lambtha = lambtha
            else:
                raise ValueError("lambtha must be a positive value")
        else:
            if (type(data) != list):
                raise ValueError("data must be a list")
            elif (len(data) < 2):
                raise ValueError("data must contain multiple values")
            else:
                total = 0
                for i in range(len(data)):
                    total += data[i]
                self.lambtha = 1 / (total / (i + 1))
    def pdf(self, x):
        if (x < 0):
            return 0

        e = 2.7182818285
        return self.lambtha * (e ** (-1 * self.lambtha * x))

    def cdf(self, x):
        if (x < 0):
            return 0

        return 1 - (self.pdf(x) / self.lambtha)
