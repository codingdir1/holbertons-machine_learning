def matrix_transpose(matrix):
    transpose = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    i = 0

    while (i < len(matrix[0])):
        j = 0
        while (j < len(matrix)):
            transpose[i][j] = matrix[j][i]
            j += 1
        i += 1
    return transpose
