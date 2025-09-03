def poly_derivative(poly):
    if (len(poly) == 0):
        return None

    power = 2
    derivative = []
    for coeff in poly[1:]:
        if ((type(coeff) != int) and (type(coeff) != float)):
            return None
        derivative.append(coeff * (power - 1))
        power += 1

    if (set(derivative) == set([0])):
        return [0]

    return derivative
