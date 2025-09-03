import numpy

def convolve_grayscale_valid(images, kernel):
    output_height = int(images.shape[1] - kernel.shape[0]) + 1
    output_width = int(images.shape[2] - kernel.shape[1]) + 1

    output = numpy.zeros((images.shape[0], output_height, output_width))
    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            output[:, i, j] = numpy.sum(images[:, i : i + kernel.shape[0], j : j + kernel.shape[1]] * kernel[None, :, :], axis=(1, 2))
    return output
