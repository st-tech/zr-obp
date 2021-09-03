import numpy as np
from scipy import integrate

from obp.ope import (
    triangular_kernel,
    epanechnikov_kernel,
    gaussian_kernel,
    cosine_kernel,
)


def test_kernel_functions():
    # triangular
    assert np.isclose(
        integrate.quad(lambda x: triangular_kernel(x), -np.inf, np.inf)[0], 1
    )
    assert np.isclose(
        integrate.quad(lambda x: x * triangular_kernel(x), -np.inf, np.inf)[0], 0
    )
    assert integrate.quad(lambda x: triangular_kernel(x) ** 2, -np.inf, np.inf)[0] > 0

    # epanechnikov
    assert np.isclose(
        integrate.quad(lambda x: epanechnikov_kernel(x), -np.inf, np.inf)[0], 1
    )
    assert np.isclose(
        integrate.quad(lambda x: x * epanechnikov_kernel(x), -np.inf, np.inf)[0], 0
    )
    assert integrate.quad(lambda x: epanechnikov_kernel(x) ** 2, -np.inf, np.inf)[0] > 0

    # gaussian
    assert np.isclose(
        integrate.quad(lambda x: gaussian_kernel(x), -np.inf, np.inf)[0], 1
    )
    assert np.isclose(
        integrate.quad(lambda x: x * gaussian_kernel(x), -np.inf, np.inf)[0], 0
    )
    assert integrate.quad(lambda x: gaussian_kernel(x) ** 2, -np.inf, np.inf)[0] > 0

    # cosine
    assert np.isclose(integrate.quad(lambda x: cosine_kernel(x), -np.inf, np.inf)[0], 1)
    assert np.isclose(
        integrate.quad(lambda x: x * cosine_kernel(x), -np.inf, np.inf)[0], 0
    )
    assert integrate.quad(lambda x: cosine_kernel(x) ** 2, -np.inf, np.inf)[0] > 0
