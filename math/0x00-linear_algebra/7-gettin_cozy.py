import copy

def cat_matrices2D(mat1, mat2, axis=0):
    matrix_shape = __import__('2-size_me_please').matrix_shape

    new_matrix = []
    if (axis == 0):
        if (matrix_shape(mat1)[1] != matrix_shape(mat2)[1]):
            return None
        new_matrix += copy.deepcopy(mat1) + copy.deepcopy(mat2)
        return new_matrix
    elif (axis == 1):
        if (matrix_shape(mat1)[0] != matrix_shape(mat2)[0]):
            return None
        for i in range(len(mat1)):
            new_matrix.append(copy.deepcopy(mat1[i]) + copy.deepcopy(mat2[i]))
        return new_matrix
