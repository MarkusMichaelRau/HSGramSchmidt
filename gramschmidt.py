import numpy as np
import hs
from hs import Vector

def inner_product(v1, v2):
    """Calculate the inner (dot) product of two vectors."""
    return hs.inner_prod(v1, v2)

def scalar_multiply(scalar, vector):
    """Multiply a vector by a scalar."""
    return hs.scal_mult(scalar, vector)

def add(u, v):
    return hs.add(u, v)

def proj(v, u): 
    nom = inner_product(v, u)
    denom = inner_product(u, u)
    return scalar_multiply(nom/denom, u)

def substract(v, u):
    minus_u = scalar_multiply(-1, u)
    return add(v, minus_u)

def gram_schmidt(vectors):
    """
    Perform Gram-Schmidt orthogonalization on a set of vectors.
    
    Parameters:
        list of vectors: List or array of Vectors of the hilbert space class.
        
    Returns:
        list of vectors: Orthogonalized vectors as rows in a list.
    """
    
    n = len(vectors)

    orthogonal = [vectors[0]]*n

    for i in range(1, n):
        projection = proj(vectors[i], orthogonal[i-1])
        orthogonal[i] = substract(vectors[i], projection)

    return orthogonal



if __name__ == "__main__":
    np.random.seed(0)
    
    grid = np.linspace(-5, 5, 1000)
    def get_normal_distribution(mean, std=1):
        return (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((grid - mean)/std)**2)

    normal_distribution_vectors = [Vector(grid, get_normal_distribution(mean)) 
                                   for mean in np.random.normal(0, 0.1, 100)]

    result = gram_schmidt(normal_distribution_vectors)
    
    print(result)
    from matplotlib import pyplot as plt
    for el in result:
        plt.plot(el.grid, np.log(el.pz))
    plt.show()


    print(inner_product(result[0], result[1]))