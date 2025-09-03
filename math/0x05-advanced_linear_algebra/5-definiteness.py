import numpy as np

def definiteness(matrix):
    if (type(matrix) != np.ndarray):
        raise TypeError("matrix must be numpy.ndarray")
    elif (len(matrix.shape) != 2 or
          matrix.shape[0] != matrix.shape[1]):
        return None

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    possibilities = [
        "Positive definite",
        "Positive semi-definite",
        "Negative semi-definite",
        "Negative definite",
        "indefinite"
                    ]

    for eigenvalue in eigenvalues:
        if (eigenvalue == 0):
            possibilities = [possibility for possibility in possibilities if possibility != "Positive definite"]
            possibilities = [possibility for possibility in possibilities if possibility != "Negative definite"]
        elif (eigenvalue > 0):
            possibilities = [possibility for possibility in possibilities if possibility != "Negative definite"]
            possibilities = [possibility for possibility in possibilities if possibility != "Negative semi-definite"]
        elif (eigenvalue < 0):
            possibilities = [possibility for possibility in possibilities if possibility != "Positive definite"]
            possibilities = [possibility for possibility in possibilities if possibility != "Positive semi-definite"]

    if ("Positive definite" in possibilities):
        return "Positive definite"
    elif ("Negative definite" in possibilities):
        return "Negative definite"
    elif ("Positive semi-definite" in possibilities):
        return "Positve semi-definite"
    elif ("Negative semi-definite" in possibilities):
        return "Negative semi-definite"
    else:
        return "idefinite"
