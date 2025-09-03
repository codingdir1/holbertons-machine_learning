def matrix_shape(matrix):
    
    if type(matrix[0]) != list:
        return list([len(matrix)])

    return list([len(matrix)] +  matrix_shape(matrix[0]))


