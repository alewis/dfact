""" Utility functions for testing matrix stuff.
"""
import jax
import jax.numpy as jnp
from jax.ops import index, index_update
import time


def dag(A):
    """
    Hermitian conjugation.
    """
    return jnp.conj(A.T)


@jax.jit
def trimultmat(A, B, C):
    """
    A @ B @ C for three dense matrices.
    """
    return jnp.dot(jnp.dot(A, B), C)


@jax.jit
def trimultdag(A, D, C):
    """
    A @ D @ C where D is a vector of diagonal entries.
    """
    return jnp.dot(A, D[:, None]*C)


def subblock_main_diagonal(A, bi=0):
    """
    Returns the indices of the elements in A's main diagonal contained in its
    [bi:, bi:] subblock.
    """
    m, n = A.shape
    idxs = jnp.arange(bi, min(m, n), dtype=jnp.int32)
    di = (idxs, idxs)
    return di


def replace_diagonal(A, D, off=0):
    """
    A is an m x n matrix.

    D is size nD <= min(m, n) representing the diagonal of an array.

    A matrix is returned identical to A, except that its first nD elements
    on the main diagonal are replaced with those of D,
    and any successive (but not preceding) diagonal elements are zeroed out.
    """
    m, n = A.shape
    k = min(m, n)
    didxs = subblock_main_diagonal(A, bi=off)
    new_elements = jnp.zeros(k, dtype=D.dtype)
    new_elements = index_update(new_elements, index[:D.size], D)
    A = index_update(A, index[didxs], new_elements)
    return A


def matshape(A):
    """
    Returns A.shape if A has two dimensions and throws a ValueError
    otherwise.
    """
    try:
        m, n = A.shape
    except ValueError:
        raise ValueError("A had invalid shape: ", A.shape)
    return (m, n)

def frob(A, B):
    """
    sqrt(sum(|A - B|**2) divided by number of elements.
    """
    assert A.size == B.size
    return jnp.sqrt(jnp.sum(jnp.abs(jnp.square(A - B))))/A.size


def gaussian_random_complex64(key=None, shape=()):
    """
    Use jax.random to generate a Gaussian random matrix of complex128 type.
    The real and imaginary parts are separately generated.
    If key is unspecified, a key is generated from system time.
    """
    if key is None:
        key = jax.random.PRNGKey(int(time.time()))
    realpart = jax.random.normal(key, shape=shape, dtype=jnp.float32)
    imagpart = jax.random.normal(key, shape=shape, dtype=jnp.float32)
    output = realpart + 1.0j * imagpart
    return output


def gaussian_random(key=None, shape=(), dtype=jnp.float32):
    """
    Generates a random matrix of the given shape and dtype, which unlike
    in pure jax may be complex. If 'key' is unspecified, a key is generated
    from system time.
    """
    if key is None:
        key = jax.random.PRNGKey(int(time.time()))

    if dtype == jnp.complex64:
        output = gaussian_random_complex64(key=key, shape=shape)
    elif dtype == jnp.complex128:
        raise NotImplementedError("double precision complex isn't supported")
    else:
        output = jax.random.normal(key, shape=shape, dtype=dtype)
    return output
