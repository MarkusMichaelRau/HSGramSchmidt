import numpy as np
from scipy.integrate import simpson as simps
from scipy.stats import lognorm
from scipy.stats import cauchy
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import numdifftools as nd
from scipy.special import eval_legendre
from scipy.stats import uniform


class HilbertSpace(object):
    """
    Implements the Aitchison Hilbert space structure for probability
    density functions (PDFs) discretized on a finite interval.

    This class provides a framework for working with a specific type of Hilbert
    space structure on the set of all PDFs defined over a finite interval.
    The operations defined—such as addition, scalar multiplication, and inner
    product—are based on the principles of Aitchison geometry, which is
    well-suited for compositional data like PDFs.

    Key Concepts:
    - Perturbation (Addition `⊕`): `f ⊕ g = C(f * g)`
    - Powering (Scalar Multiplication `⊙`): `a ⊙ f = f^a`
    - Closure (`C`): Normalization to ensure the result integrates to 1.
    """

    def __init__(self, num):
        """
        Initializes the Hilbert space by defining the discrete grid.

        Parameters
        ----------
        num : int
            The number of points for discretizing the interval.
        """
        self.l = 1.0
        self.grid = np.linspace(-self.l, self.l, num=num)

    def C(self, g):
        """
        The closure operation. Normalizes a function to be a valid PDF.

        Normalizes a function `g` so that it integrates to 1 over the defined
        grid, ensuring it is a valid probability density function.

        Formula: C(g(x)) = g(x) / ∫[g(t) dt]

        Parameters
        ----------
        g : np.ndarray
            A function evaluated on the grid.

        Returns
        -------
        np.ndarray
            The normalized function.
        """
        return g / simps(g, x=self.grid)

    def scal_mult(self, alpha, g):
        """
        Performs scalar multiplication in the Aitchison sense (powering).

        Formula: (α ⊙ g)(x) = g(x)^α

        Parameters
        ----------
        alpha : float
            The scalar multiplier.
        g : np.ndarray
            The function to be multiplied.

        Returns
        -------
        np.ndarray
            The resulting function.
        """
        return np.power(g, alpha)

    def add(self, f, g):
        """
        Performs addition in the Aitchison sense (perturbation).

        This is the closure of the element-wise product of two functions.

        Formula: (f ⊕ g)(x) = C(f(x) * g(x))

        Parameters
        ----------
        f : np.ndarray
            The first function.
        g : np.ndarray
            The second function.

        Returns
        -------
        np.ndarray
            The "sum" of the two functions.
        """
        return self.C(f * g)

    def inner_prod(self, f, g):
        """
        Calculates the Aitchison inner product between two functions.

        The inner product is defined based on the logarithms of the functions.
        Formula: ⟨f, g⟩ₐ = ∫[ln(f)ln(g)dx] - (1/2l) * ∫[ln(f)dx] * ∫[ln(g)dx]

        Parameters
        ----------
        f : np.ndarray
            The first function.
        g : np.ndarray
            The second function.

        Returns
        -------
        float
            The scalar value of the inner product.
        """
        first_term = simps(np.log(f) * np.log(g), x=self.grid)
        second_term = (1. / (2 * self.l)) * simps(np.log(f), x=self.grid) * simps(np.log(g), x=self.grid)
        return first_term - second_term

    def norm(self, f):
        """
        Calculates the Aitchison norm of a function.

        The norm is induced by the inner product: norm(f) = sqrt(inner_prod(f, f)).
        Formula: ||f||ₐ = sqrt(⟨f, f⟩ₐ)

        Parameters
        ----------
        f : np.ndarray
            The function.

        Returns
        -------
        float
            The norm of the function.
        """
        first_term = simps(np.log(f) * np.log(f), x=self.grid)
        second_term = simps(np.log(f), x=self.grid)**2
        completed = first_term - (1. / (2 * self.l)) * second_term
        return np.sqrt(completed)

    def distance(self, f, g):
        """
        Calculates the Aitchison distance between two functions.

        The distance is the norm of the Aitchison difference between f and g.
        Formula: d(f, g) = ||f ⊖ g||ₐ

        Parameters
        ----------
        f : np.ndarray
            The first function.
        g : np.ndarray
            The second function.

        Returns
        -------
        float
            The distance between f and g.
        """
        return np.sqrt(self.inner_prod(f, f) - 2 * self.inner_prod(f, g) + self.inner_prod(g, g))

    def check(self, f):
        """
        Verifies if a function `f` is defined on a compatible grid.

        This utility function assumes `f` is an object that has a `.grid`
        attribute for comparison.

        Parameters
        ----------
        f : object
            An object with a `.grid` attribute.

        Raises
        ------
        ValueError
            If the grids are not identical.

        Returns
        -------
        int
            Returns 0 if the check passes.
        """
        if not np.isclose(np.sum(f.grid - self.grid), 0.):
            raise ValueError('Incompatible pdf')
        else:
            return 0

    def legendre_basis(self, n):
        """
        Generates the n-th orthonormal basis function for this Hilbert space.

        The basis is constructed by transforming Legendre polynomials into the
        Aitchison space.

        Parameters
        ----------
        n : int
            The non-negative integer order of the basis function.

        Returns
        -------
        np.ndarray
            The n-th basis function as a normalized PDF on the grid.
        """
        x = self.grid
        legendre_poly = eval_legendre(n, x)
        scaling_factor = np.sqrt((2 * n + 1) / (2 * self.l))
        return self.C(np.exp(scaling_factor * legendre_poly))

    def legendre_aich(self, n):
        """
        Calculates the coordinate representation of the n-th Legendre basis.

        This is the scaled Legendre polynomial itself, representing a vector
        in the log-transformed coordinate system.

        Parameters
        ----------
        n : int
            The non-negative integer order of the polynomial.

        Returns
        -------
        np.ndarray
            The scaled n-th Legendre polynomial evaluated on the grid.
        """
        x = self.grid
        return np.sqrt((2 * n + 1) / (2 * self.l)) * eval_legendre(n, x)


class PDF(object):
    """
    Represents a Probability Density Function (PDF) on a discrete grid.

    This class provides tools to work with PDFs within the Aitchison
    Hilbert space framework. It allows for the representation of a PDF as a
    linear combination of Legendre basis functions and can reconstruct a
    PDF from a given set of coefficients.

    Attributes
    ----------
    mids : np.ndarray
        The discrete grid points on which the PDF is defined.
    pdf : np.ndarray
        The normalized probability density values at each grid point.
    hs : HilbertSpace
        An instance of the HilbertSpace class, configured with the same grid
        as the PDF, providing the necessary Aitchison geometry tools.
    """

    def __init__(self, grid, pdf):
        """
        Initializes and normalizes a PDF object.

        Parameters
        ----------
        grid : np.ndarray
            A 1D array of points representing the domain of the PDF. The number
            of points must be odd.
        pdf : np.ndarray
            A 1D array of the PDF values corresponding to the grid points.
            The values will be normalized to ensure the PDF integrates to 1.

        Raises
        ------
        ValueError
            If the number of grid points is not odd.
        """
        if len(grid) % 2 == 0:
            raise ValueError('Need an odd number of grid points for initialization.')

        self.mids = grid
        # Normalize the incoming PDF using the trapezoidal rule for integration
        self.pdf = pdf / np.trapz(pdf, self.mids)
        self.hs = HilbertSpace(len(self.mids))

    def represent(self, num_coeff):
        """
        Finds the Legendre basis coefficients that represent the PDF.

        This method performs an optimization to find a set of coefficients
        for the first `num_coeff` Legendre basis functions (in the Aitchison
        space) that best approximate the logarithm of the current PDF. The
        approximation is based on minimizing the mean squared error.

        Parameters
        ----------
        num_coeff : int
            The number of Legendre coefficients to use for the representation.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The result object from `scipy.optimize.minimize`. The optimal
            coefficients can be accessed via the `.x` attribute of this object.
        """
        # Define the loss function as the mean squared error between the
        # reconstructed log-PDF and the actual log-PDF.
        def loss(coeffs):
            reconstructed_log_pdf = np.sum(
                [coeffs[j] * self.hs.legendre_aich(j) for j in range(num_coeff)],
                axis=0
            )
            return np.mean((reconstructed_log_pdf - np.log(self.pdf))**2)

        # Initial guess for the optimizer.
        # Note: A more robust approach might be np.zeros(num_coeff).
        initial_guess = np.zeros(num_coeff)
        if num_coeff == 5:
            initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1]


        res = minimize(loss, x0=initial_guess)
        return res

    def evaluate(self, coefs):
        """
        Reconstructs a PDF from a given set of Legendre coefficients.

        This method takes a set of coefficients and generates the
        corresponding PDF by performing a weighted sum of the Legendre basis
        functions in the log-space, exponentiating the result, and finally
        normalizing it.

        Parameters
        ----------
        coefs : array_like
            A list or array of coefficients for the Legendre basis expansion.

        Returns
        -------
        np.ndarray
            The reconstructed and normalized PDF evaluated on the grid.
        """
        # Reconstruct the log-PDF from the coefficients
        log_output_pdf = np.sum(
            [coefs[j] * self.hs.legendre_aich(j) for j in range(len(coefs))],
            axis=0
        )
        # Exponentiate to get the PDF and normalize
        output_pdf = np.exp(log_output_pdf)
        return output_pdf / np.trapz(output_pdf, self.mids)
