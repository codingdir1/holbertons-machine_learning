class Poisson():
    def __init__(self, data = None, lambtha = 1.):

        if (data == None):
            if (lambtha <= 0):
                raise ValueError("lambtha must be positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if (type(data) != list):
                raise ValueError("data must be a list")
            elif (len(data) < 2):
                raise ValueError("data must contain multiple values")
            else:
                total = 0
                for i in range(len(data)):
                    total += data[i]
                self.lambtha = float(total / (i + 1))

    def pmf(self, k):
        if (k < 0):
            return 0

        k = int(k)
        e = 2.7182818285
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        return ((self.lambtha) ** k) * (e ** (-1 * self.lambtha)) / (factorial)

    def cdf(self, k):
        if (k < 0):
            return 0

        k = int(k)
        total = 0
        for i in range(k + 1):
            total += self.pmf(i)
        return total
