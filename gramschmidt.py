import numpy as np
from hs import *

def inner_product(v1, v2):
    """Calculate the inner (dot) product of two vectors."""
    return np.dot(v1, v2)

def scalar_multiply(scalar, vector):
    """Multiply a vector by a scalar."""
    return scalar * vector


class Vector: 
    def __init__(self, )

def gram_schmidt(vectors):
    """
    Perform Gram-Schmidt orthogonalization on a set of vectors.
    
    Parameters:
        vectors (list or np.ndarray): List or array of vectors (each vector is an array).
        
    Returns:
        np.ndarray: Orthogonalized vectors as rows in a 2D numpy array.
    """
    vectors = np.array(vectors, dtype=float)
    n = vectors.shape[0]

    orthogonal = np.zeros_like(vectors)

    for i in range(n):
        vec = vectors[i]
        for j in range(i):
            proj_scalar = inner_product(orthogonal[j], vec) / inner_product(orthogonal[j], orthogonal[j])
            proj = scalar_multiply(proj_scalar, orthogonal[j])
            vec = vec - proj
        orthogonal[i] = vec

    return orthogonal

# Example usage:
vectors = [
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
]

orthogonal_vectors = gram_schmidt(vectors)
print("Orthogonal vectors:")
print(orthogonal_vectors)


