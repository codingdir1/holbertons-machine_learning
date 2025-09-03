def determinant(matrix):
    if (matrix == [[]]):
        return 1

    if (type(matrix) != list or
        (type(matrix) == list and len(matrix) == 0) or
        (type(matrix) == list and type(matrix[0]) != list)): 
        raise TypeError("matrix must be a list of lists")
    for j in range(len(matrix)):
        if (len(matrix) != len(matrix[j])):
            raise TypeError("matrix must be a square matrix")
    
    result = 0
    for i in range(len(matrix[0])):
        sign = (-1) ** i
        result += sign * matrix[0][i] * minor_I_J(matrix, 0, i)
    return result

def minor_I_J(matrix, i,  j):
    new_matrix = []
    for _i in range(len(matrix)):
        if (len(new_matrix) == 0 or new_matrix[-1] != []): 
            new_matrix.append([])
        for _j in range(len(matrix[_i])):
            if (_i != i and _j != j):
                        new_matrix[-1].append(matrix[_i][_j])

    # remove mis-added empty-arrays from new_matrix
    if (new_matrix[-1] == [] and len(new_matrix) > 1):
        new_matrix.pop(-1)
    return determinant(new_matrix)
