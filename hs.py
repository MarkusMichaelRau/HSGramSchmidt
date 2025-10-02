
import numpy as np
from scipy.integrate import simpson as simps

def normalize(grid, pz): 
    return pz/simps(pz, grid)

class Vector:
    def __init__(self, grid, pz):
        self.grid = grid
        self.pz = normalize(grid, pz)
        self.l = (grid[-1] - grid[0])/2.

def C(grid, pz):   
    return Vector(grid, normalize(grid, pz))

def scal_mult(alpha, g):
    return C(g.grid, np.power(g.pz, alpha))

def add(f, g):
    assert np.all(g.grid == f.grid)
    return C(f.grid, f.pz * g.pz)

def inner_prod(f, g):
    print(f)
    assert np.all(f.grid == g.grid)
    first_term = simps(np.log(f.pz) * np.log(g.pz), x=f.grid)
    second_term = (1. / (2 * f.l)) * simps(np.log(f.pz), x=f.grid) * simps(np.log(g.pz), x=g.grid)
    return first_term - second_term

def norm(f):
        
    first_term = simps(np.log(f.pz) * np.log(f.pz), x=f.grid)
    second_term = simps(np.log(f.pz), x=f.grid)**2
    completed = first_term - (1. / (2 * f.l)) * second_term
    return np.sqrt(completed)

def distance(f, g):
    assert np.all(f.grid == g.grid)
    return np.sqrt(inner_prod(f, f) - 2. * inner_prod(f, g) + inner_prod(g, g))


if __name__ == "__main__":
    grid = np.linspace(-5, 5, 1000)
    def get_normal_distribution(mean, std=1):
        return (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((grid - mean)/std)**2)

    normal_distribution_vectors = [Vector(grid, get_normal_distribution(mean)) 
                                   for mean in np.random.normal(0, 1, 10)]

    print(distance(normal_distribution_vectors[0], normal_distribution_vectors[1]))
    print(inner_prod(normal_distribution_vectors[0], normal_distribution_vectors[1]))
    print(norm(normal_distribution_vectors[0]))
    print(add(normal_distribution_vectors[0], normal_distribution_vectors[1]))
    print(scal_mult(2, normal_distribution_vectors[0]))
