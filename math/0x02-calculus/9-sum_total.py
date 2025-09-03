def summation_i_squared(n):
    if (n < 1 or type(n) != int):
        return None
    
    total = 0

    i = 1
    while (i <= n):
        total += i * i
        i += 1

    return total
