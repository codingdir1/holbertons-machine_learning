def np_slice(matrix, axes={}):
    sliced = [slice(None)] * (max(axes) + 1)

    for axis, value in axes.items():
        sliced[axis] = slice(*value)

    new_matrix = matrix[tuple(sliced)]
    return new_matrix
