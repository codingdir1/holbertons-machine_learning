def poly_integral(poly, C=0):
    if (len(poly) == 0):
        return None

    if ((type(C) != int) and (type(C) != float)):
        return None

    integral = [C]
    power = 0
    for coeff in poly:
        if ((type(coeff) != int) and (type(coeff) != float)):
            return None
        
        integral_coeff = coeff / (power + 1)
        if (integral_coeff == int(integral_coeff)):
            integral.append(int(integral_coeff))
        else:
            integral.append(integral_coeff)

        power += 1

    return integral
