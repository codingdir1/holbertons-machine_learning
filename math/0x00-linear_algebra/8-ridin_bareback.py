def mat_mul(mat1, mat2):
    matrix_shape = __import__('2-size_me_please').matrix_shape

    if (mat1 == [] or mat2 == []):
        return None

    if (matrix_shape(mat1)[1] != matrix_shape(mat2)[0]):
        return None

    matrix_product = [[0 for j in range(matrix_shape(mat2)[1])] 
                      for i in range(matrix_shape(mat1)[0])]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat1[0])):
                matrix_product[i][j] += mat1[i][k] * mat2[k][j]

    return matrix_product

