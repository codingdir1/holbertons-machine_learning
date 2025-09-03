determinant = __import__("0-determinant").determinant
adjugate = __import__("3-adjugate").adjugate

def inverse(matrix):
    if (type(matrix) != list or
        (type(matrix) == list and len(matrix) == 0) or
        (type(matrix) == list and type(matrix[0]) != list)): 
        raise TypeError("matrix must be a list of lists")
    if (len(matrix) == 0):
        raise TypeError("matrix must be a non-empty square matrix")
    for j in range(len(matrix)):
        if (len(matrix) != len(matrix[j])):
            raise TypeError("matrix must be a non-empty square matrix")

    det_matrix = determinant(matrix)
    if (det_matrix == 0):
        return None

    inv_matrix = adjugate(matrix)
    for i in range(len(inv_matrix)):
        for j in range(len(inv_matrix[i])):
            inv_matrix[i][j] /= det_matrix
    return inv_matrix

