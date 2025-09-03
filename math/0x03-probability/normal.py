class Normal():
    def __init__(self, data = None, mean = 0., stddev = 1.):
        self.e = 2.7182818285
        self.pi = 3.1415926536
        if (data == None):
            if (stddev < 0):
                raise ValueError("stddev mst be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if (type(data) != list):
                raise ValueError("data must be list")
            elif (len(data) < 2):
                raise ValueError("data must contain multiple values")
            else:
                #calculating mean
                total = 0
                for i in range(len(data)):
                    total += data[i]
                self.mean = float(total / (i + 1))

                #calclating standard deviation
                total = 0
                for i in range(len(data)):
                    total += (self.mean - data[i]) ** 2
                self.stddev = float((total / (i + 1)) ** 0.5)
    
    def z_score(self, x):
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        return (self.e ** ((-1 * (x - self.mean) ** 2) / (2 * self.stddev ** 2))) / (self.stddev * (2 * self.pi) ** 0.5)

    def factorial(self, n):
        product = 1
        i = 1
        while (i <= n):
            product *= i
            i += 1
        return product

    def erf(self, z):
        summation = 0
        for i in range(5):
            summation += ((-1) ** i) * (z ** (2 * i + 1)) / (self.factorial(i) * (2 * i + 1))

        return (2 / ((self.pi) ** 0.5)) * summation


    def cdf(self, x):
        return 0.5 * (1 + self.erf((x - self.mean) / (self.stddev * (2 ** 0.5))))
