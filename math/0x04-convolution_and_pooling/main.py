#!/usr/bin/env python3

conv2D = __import__('conv').conv2D

image = [
            [[1, 9], [2, 8], [3, 7]],
            [[4, 6], [5, 5], [6, 4]],
            [[7, 3], [8, 2], [9, 1]]
        ]

#kernel = [
#            [[[1, 0],  [0, 1]], [[0, 1],  [1, 0]]],
#            [[[-1, 1], [0, 1]], [[0, 0], [-1, 0]]]
#          ]

kernel1 = [
            [[1,  0], [0,  1]],
            [[-1, 0], [0, -1]]
          ]
kernel2 = [
            [[0, 1], [1,  0]],
            [[1, 1], [0,  0]]
          ]

print(conv2D(image, kernel1))
print(conv2D(image, kernel2))
