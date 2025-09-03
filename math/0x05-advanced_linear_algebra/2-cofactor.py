minor_I_J = __import__("0-determinant").minor_I_J

def cofactor(matrix): 
    if (type(matrix) != list or
        (type(matrix) == list and len(matrix) == 0) or
        (type(matrix) == list and type(matrix[0]) != list)): 
        raise TypeError("matrix must be a list of lists")
    if (len(matrix) == 0):
        raise TypeError("matrix must be a non-empty square matrix")
    for j in range(len(matrix)):
        if (len(matrix) != len(matrix[j])):
            raise TypeError("matrix must be a non-empty square matrix")
    
    new_matrix = []
    for i in range(len(matrix)):
        new_matrix.append([])
        for j in range(len(matrix[i])):
            new_matrix[-1].append(((-1) ** (i + j)) * minor_I_J(matrix, i, j))
    return new_matrix
