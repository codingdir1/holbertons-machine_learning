class Binomial():
    def __init__(self, data = None, n = 1, p = 0.5):
        if (data == None):
            if (n <= 0):
                raise ValueError("n must be a positive value")
            elif (p <= 0 or p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = n
                self.p = p
        else:
            if (type(data) != list):
                raise ValueError("data must be a list")
            elif (len(data) < 2):
                raise ValueError("data must contain multiple values")
            else:
                 total = 0
                 #mean
                 for i in range(len(data)):
                     total += data[i]
                 mean = total / (i + 1)

                 #variance
                 total = 0
                 for i in range(len(data)):
                     total += (data[i] - mean) ** 2
                 variance = total / (i + 1)

                 p = 1 - variance / mean
                 self.n = round(mean / p)
                 self.p = mean / self.n
    
    def factorial(self, n):
        product = 1
        while (n > 0):
            product *= n
            n -= 1
        return product

    def pmf(self, k):
        if (k < 0):
            return 0
        k = int(k)
        return (self.factorial(self.n)) * (self.p ** k) * ((1 - self.p) ** (self.n - k)) / (self.factorial(k) * self.factorial(self.n - k))

    def cdf(self, k):
        if (k < 0):
            return 0

        k = (int(k))
        total = 0
        for i in range(k + 1):
            total += self.pmf(i)
        return total
