def add_arrays(arr1, arr2):
    matrix_shape = __import__('2-size_me_please').matrix_shape

    if (matrix_shape(arr1) != matrix_shape(arr2)):
        return None

    sum_array = []
    for i in range(len(arr1)):
        sum_array.append(arr1[i] + arr2[i])
    return sum_array
