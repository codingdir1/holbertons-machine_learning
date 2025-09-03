def add_matrices2D(mat1, mat2):
    matrix_shape = __import__('2-size_me_please').matrix_shape

    shape = matrix_shape(mat1)
    if (shape != matrix_shape(mat2)):
        return None

    new_matrix = [[0 for _ in range(shape[1])] for _ in range(shape[0])]

    i = 0
    for i in range(shape[0]):
        j = 0
        for j in range(shape[1]):
            new_matrix[i][j] = mat1[i][j] + mat2[i][j]
    return new_matrix
