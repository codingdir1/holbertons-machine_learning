def add_matrices(mat1, mat2):
    matrix_shape = __import__('2-size_me_please').matrix_shape

    if (matrix_shape(mat1) != matrix_shape(mat2)):
        return None

    new_matrix = []
    length = len(mat1)
    if (type(mat1[0]) == int or type(mat1[0]) == float):
        new_matrix = [0 for _ in range(length)]
        for i in range(length):
            new_matrix[i] = mat1[i] + mat2[i]
        return new_matrix
    else:
        new_matrix = [[] for _ in range(length)]
        for i in range(length):
            new_matrix[i] = add_matrices(mat1[i], mat2[i])
        return new_matrix
