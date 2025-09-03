# function for 2D convolution
def conv2D(image, kernel, padding = 'valid', stride = (1, 1)):

    # checking data types
    if (type(image) != list or type(kernel) != list):
        raise ValueError("The image or the kernel is not a list")
    elif (type(image[0]) != list or type(kernel[0]) != list):
        raise ValueError("The image or the kernel is 1D list")
    
    # checking sizes
    if (len(kernel) > len(image) or
        len(kernel[0]) > len(image[0])):
        raise ValueError("The kernel is too large")
    elif (type(image[0][0]) == list):
        if (len(image[0][0]) != len(kernel[0][0])):
            raise ValueError("Mismatch in the number of channels")
    
    n_kernels = 0
    #checking for multi-kernels
    if (type(kernel[0][0][0]) == list):
        n_kernels = len(kernel[0][0][0])

    # dealing with padding issue
    if (padding == 'same'):
        p_x, p_y = (len(kernel[0]) - 1) / 2, (len(kernel) - 1) / 2
        for i in range(int(p_y)):
            image.insert(0, [0 for _ in range(len(image[0]))])
            image.append([0 for _ in range(len(image[0]))])
        for i in range(len(image)):
            for j in range(int(p_x)):
                image[i].insert(0, 0)
                image[i].append(0)
    elif (padding != 'valid'):
        raise ValueError("Invalid padding mode")

    conv_output_w = int((len(image[0]) - len(kernel[0])) / stride[1]) + 1
    conv_output_h = int((len(image) - len(kernel)) / stride[0]) + 1

    if (n_kernels == 0):
        conv_output = []
    
        for i in range(0, (len(image) - len(kernel)) + 1, stride[0]):
            conv_output.append([])
            for j in range(0, (len(image[0]) - len(kernel)) + 1, stride[1]):
                conv_output[-1].append(elementwise_prod(image, kernel, i, j))
    
        return conv_output
    else:
        conv_outputs = []

        for h in range(n_kernels):
            conv_output = []
    
            for i in range(0, (len(image) - len(kernel)) + 1, stride[0]):
                conv_output.append([])
                for j in range(0, (len(image[0]) - len(kernel)) + 1, stride[1]):
                    conv_output[-1].append(elementwise_prod(image, kernel, i, j, h))
            conv_outputs.append(conv_output)
        return conv_outputs

# function for elemntwise product of matricies with
# offset i and j on image
def elementwise_prod(image, kernel, i, j, n_kernel = None):
    width = len(kernel[0])
    height = len(kernel)
    n_channels = 0
    if (type(image[0][0]) == list):
        n_channels = len(kernel[0][0])

    total = 0
    if (n_kernel == None):
        for _i in range(i, i + height):
            for _j in range(j, j + width):
                if (n_channels == 0):
                    if ((type(image[_i][_j]) != int and type(image[_i][_j]) != float) or
                        (type(kernel[_i - i][_j - j]) != int and type(kernel[_i - i][_j - j]) == float)):
                        raise ValueError("Non-number entry in the image or the kernel")
                    else:
                        total += image[_i][_j] * kernel[_i - i][_j - j]
                else:
                    for _k in range(n_channels):
                        if ((type(image[_i][_j][_k]) != int and type(image[_i][_j][_k]) != float) or
                            (type(kernel[_i - i][_j - j][_k]) != int and type(kernel[_i - i][_j - j][_k]))):
                            raise ValueError("Non-number entry in the image or the kernel")
                        else:
                            total += image[_i][_j][_k] * kernel[_i - i][_j - j][_k]
        return total
    else:
        for _i in range(i, i + height):
            for _j in range(j, j + width):
                if (n_channels == 0):
                    if ((type(image[_i][_j]) != int and type(image[_i][_j]) != float) or
                        (type(kernel[_i - i][_j - j][n_kernel]) != int and type(kernel[_i - i][_j - j][n_kernel]) != float)):
                        raise ValueError("Non-number entry in the image or the kernel")
                    else:
                        total += image[_i][_j] * kernel[_i - i][_j - j][n_kernel]
                else:
                    for _k in range(n_channels):
                        if ((type(image[_i][_j][_k]) != int and type(image[_i][_j][_k]) != float) or
                            (type(kernel[_i - i][_j - j][_k][n_kernel]) != int and type(kernel[_i - i][_j - j][_k][n_kernel]) != float)):
                            raise ValueError("Non-number entry in the image or the kernel")
                        else:
                            total += image[_i][_j][_k] * kernel[_i - i][_j - j][_k][n_kernel]
        return total
