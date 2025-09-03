def cat_matrices(mat1, mat2, axis=0):
    matrix_shape = __import__('2-size_me_please').matrix_shape
    new_matrix = []

    if (type(mat1[0]) == int or type(mat1[0]) == float):
        if (axis == 0):
            new_matrix += mat1
            new_matrix += mat2
            return new_matrix
        else:
            return None

    if (matrix_shape(mat1) != matrix_shape(mat2)):
        return None


      


