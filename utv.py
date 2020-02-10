""" Code to implement the 'UTV' decomposition, and to compare its accuracy
with the SVD.

Adam GM Lewis
adam.lws@gmail.com
alewis@perimeterinstitute.ca

Will probably borrow heavily from cod by Per-Gunnar Martinsson,
Gregorio Quintana-Orti, and Nathan Heavner at https://github.com/flame/randutv.
"""


import jax
from jax.ops import index_update, index
from jax.lax import cond
import jax.numpy as jnp
import numpy as np
import math
from functools import partial

import matutils
from matutils import dag

import qr


def svd_truncation(A, b=None):
    """
    Returns the SVD of A, truncated to b singular values. If b is None,
    no truncation occurs. Also returns the truncation error 'eps' (sum of the
    squared truncated singular values).

    Returns: [U, S, V, eps]
    """
    if b is None:
        b = A.shape[1]
    assert b <= A.shape[1]

    U, S, V = jnp.linalg.svd(A, full_matrices=False)
    Utrunc = U[:, :b]
    Strunc = S[:b]
    eps = jnp.sum(S[b:]**2)
    Vtrunc = V[:b, :]
    output = [Utrunc, Strunc, Vtrunc, eps]
    return output


def rand_range_col(A, b=None, n_iter=1, full=True):
    """
    Implements randomized range finding. Starting from a general (m x n) matrix
    A, a (m x b) with b<=n matrix, Q, is sought such that
    ||I - Q Q^dag A|| is minimized.

    It is fairly straightforward to modify this algorithm so that k is
    instead chosen dynamically in such a way that the error function is
    bound by some chosen epsilon. I haven't done this because the code is
    slightly more complicated (the QR algorithm must be written out and
    slightly modified).
    This function returns a (m x b) matrix.

    Arguments
    ---------
    A (numpy array): The m x n matrix to be factorized.
    b (int)        : Rank of the output. b=n if unspecified or if b>n.
    n_iter (int)   : Number of power iterations

    Returns
    -------
    The m x b matrix Q s.t. A ~ Q Q^dag A.
        A point of confusion: Q is supposed to be unitary, so isn't
                              Q Q^dag = I?

    Exceptions
    ----------
    ValueError when A is not a two-dimensional matrix.
    AssertionError unless k > 0, p>=0, n_iter>=0.

    """
    m, n = matutils.matshape(A)

    if b is None or b > n:
        b = n

    assert b > 0
    assert n_iter >= 0

    R = matutils.gaussian_random(shape=(n, b), dtype=A.dtype)
    Y = jnp.dot(A, R)  # (m x b) projection of A onto that subspace.

    if n_iter > 0:  # Power iterations speed the decay of Y's singular values.
        AAdag = jnp.dot(A, dag(A))
        for _ in range(n_iter):
            Y = jnp.dot(AAdag, Y)

    mode = "reduced"
    if full:
        mode = "complete"
    Q, _ = jnp.linalg.qr(Y)  # Now Q is unitary with the same column space as Y.
    return Q


def rand_range_row(A, b=None, n_iter=1, mode="reduced"):
    """
    Implements randomized range finding. Starting from a general (m x n) matrix
    A, a (n x b) with b<=m matrix, Q, is sought such that
    ||I - Q Q^dag A|| is minimized.

    It is fairly straightforward to modify this algorithm so that k is
    instead chosen dynamically in such a way that the error function is
    bound by some chosen epsilon. I haven't done this because the code is
    slightly more complicated (the QR algorithm must be written out and
    slightly modified).

    If full=False:
        This function returns a (n x b) matrix.

    If full=True:
        This function returns a (n x n) matrix.

    Arguments
    ---------
    A (numpy array): The m x n matrix to be factorized.
    b (int)        : Rank of the output. b=n if unspecified or if b>n.
    n_iter (int)   : Number of power iterations

    Returns
    -------
    The n x b matrix (full=False) or n x n matrix (full=True)
    Q s.t. A ~ Q Q^dag A.
        A point of confusion: Q is supposed to be unitary, so isn't
                              Q Q^dag = I?

    Exceptions
    ----------
    ValueError when A is not a two-dimensional matrix.
    AssertionError unless k > 0, p>=0, n_iter>=0.

    """
    m, n = A.shape
    b = cond(b is None or b > n,
             n, lambda x: x,
             b, lambda x: x)

    # assert n_iter >= 0
    # G = jnp.random.randn(m, b)  # The Gaussian random subspace.
    G = matutils.gaussian_random(shape=(m, b), dtype=A.dtype)
    Y = jnp.dot(dag(A), G)  # (n x b) projection of A onto that subspace.

    # if n_iter > 0:
    # Power iterations speed the decay of Y's singular values.
    AdagA = jnp.dot(dag(A), A)
    for _ in range(n_iter):
        Y = jnp.dot(AdagA, Y)

    Qout = qr.house_qr(Y, mode=mode)
    # Now Q is unitary with the same column space as Y.
    return Qout


@jax.jit
def rand_range_row_jit(A, G, q=2):
    """
    Jit implementation of
    randomized range finding. Starting from a general (m x n) matrix
    A, a (n x b) with b<=m matrix, Q, is sought such that
    ||I - Q Q^dag A|| is minimized.

    It is fairly straightforward to modify this algorithm so that k is
    instead chosen dynamically in such a way that the error function is
    bound by some chosen epsilon. I haven't done this because the code is
    slightly more complicated (the QR algorithm must be written out and
    slightly modified).

    If full=False:
        This function returns a (n x b) matrix.

    If full=True:
        This function returns a (n x n) matrix.

    Arguments
    ---------
    A (array): The m x n matrix to be factorized.
    G (array): An m x b array that will be overwritten with random
               numbers.
    n_iter (int)   : Number of power iterations

    Returns
    -------
    The n x b matrix (full=False) or n x n matrix (full=True)
    Q s.t. A ~ Q Q^dag A.
        A point of confusion: Q is supposed to be unitary, so isn't
                              Q Q^dag = I?

    Exceptions
    ----------
    ValueError when A is not a two-dimensional matrix.
    AssertionError unless k > 0, p>=0, n_iter>=0.

    """
    m, n = A.shape
    # assert n_iter >= 0
    # G = jnp.random.randn(m, b)  # The Gaussian random subspace.
    G = index_update(G, index[:], matutils.gaussian_random_fill(G))
    Y = dag(A) @ G  # (n x b) projection of A onto that subspace.

    # if n_iter > 0:
    # Power iterations speed the decay of Y's singular values.
    # TODO How can we loop a dynamical number of times??
    AdagA = dag(A) @ A
    #for _ in range(q):
    for _ in range(2):
        Y = index_update(Y, index[:], AdagA@Y)

    Qout = qr.house_qr(Y, mode="WY")
    # Now Q is unitary with the same column space as Y.
    return Qout




def randSVD(A, k=None, p=5, n_iter=2):
    """
    Implements the 'randSVD' algorithm, approximating the full SVD of A
    via random sampling methods. I wrote this following the version in the
    randUTV talk.

    Arguments
    ---------
    A (numpy array): The m x n matrix to be factorized.
    k (int)        : Rank of the output. k=n if unspecified or if k>n.
    p (int)        : Oversampling parameter. In the throughput, we take
                     k -> k + p. Larger values entail better, but
                     slower, results.
    n_iter (int)   : Number of power iterations


    Exceptions
    ----------
    ValueError when A is not a two-dimensional matrix.
    AssertionError unless k > 0, p>=0, n_iter>=0.

    Returns
    -------
    List [U (m x k), S (k), Vs (k x n)] where
        A ~ U * diag(S) * Vs after padding k back to n.
    """
    try:
        m, n = A.shape
    except ValueError:
        raise ValueError("A had invalid shape: ", A.shape)

    if k is None or k > n:
        k = n

    assert k > 0
    assert p >= 0
    assert n_iter >= 0

    Q = rand_range_col(A, b=k+p, n_iter=n_iter)
    B = jnp.dot(dag(Q), A)
    Utilde, S, Vdag = jnp.linalg.svd(B, full_matrices=False)
    U = jnp.dot(Q, Utilde)

    U = U[:, :k]
    S = S[:k]
    Vdag = Vdag[:k, :]
    output = [U, S, Vdag]
    return output


# Transcribing the "slow" code in Figure 3 of the randUTV paper
# into numpy (or jax.numpy).
def randUTV_slow(A, b, q):
    T = A
    m, n = A.shape
    U = jnp.eye(m, dtype=A.dtype)
    V = jnp.eye(n, dtype=A.dtype)
    for i in range(math.ceil(n/b)):
        bidx = b*i
        Tb = T[bidx:, bidx:]
        if n - bidx > b:
            UU, TT, VV = stepUTV_slow(Tb, b=b, n_iter=q)

        else:
            UU, TTs, VVh = jnp.linalg.svd(Tb, full_matrices=True)
            VV = dag(VVh)
            TT = jnp.zeros(Tb.shape, A.dtype)
            TTd = jnp.diag(TTs)
            TT = index_update(TT, index[:TTd.shape[0], :TTd.shape[1]],
                              TTd)
        U = index_update(U, index[:, bidx:], jnp.dot(U[:, bidx:], UU))
        V = index_update(V, index[:, bidx:], jnp.dot(V[:, bidx:], VV))
        T = index_update(T, index[bidx:, bidx:], TT)
        T = index_update(T, index[:bidx, bidx:], jnp.dot(T[:bidx, bidx:], VV))
    return [U, T, V]


def stepUTV_slow(A, b=None, p=5, n_iter=1, verbose=False):
    """
    Perfoms one step of the randUTV algorithm using the 'slow' method
    of Figure 3.

    This algorithm applies the UTV decomposition to one block of size
    b. If b is None, the entire matrix is decomposed.

    Arguments
    ---------
    A (numpy array): The m x n matrix to be factorized.
    b (int)        : Block size of the output.
    p (int)        : Oversampling parameter.
    n_iter (int)   : Number of power iterations for the Gaussian sampling.


    Exceptions
    ----------
    ValueError when A is not a two-dimensional matrix.
    AssertionError unless b > 0, p>=0, n_iter>=0.

    Returns
    -------
    List [U (m x m), T (m x n), dag(V) (n x n)] where
        A = U @ T @ dag(V) .
    """

    try:
        m, n = A.shape
    except ValueError:
        raise ValueError("A had invalid shape: ", A.shape)

    if b is None or b > n:
        b = n

    if m < n:
        raise NotImplementedError("m < n case of stepUTV_slow not implemented.")
    #assert m >= n
    assert b > 0
    assert p >= 0
    assert n_iter >= 0

    V, _ = rand_range_row(A, b=b+p, n_iter=n_iter, mode="complete")  # (n x n)

    # First b columns approximately span the singular value space of
    # A.
    AV = jnp.dot(A, V)
    AV1 = jnp.dot(A, V[:, :b])
    AV2 = jnp.dot(A, V[:, b:])
    U, T11, Vsmall_dH = jnp.linalg.svd(AV1)  # (m x m), min(m, b), (b x b)
    Vsmall_d = dag(Vsmall_dH)

    Tright = jnp.dot(dag(U), AV2)
    T = jnp.zeros((m, n), dtype=A.dtype)
    T = jax.ops.index_update(T, jax.ops.index[:b, :b], jnp.diag(T11))
    T = jax.ops.index_update(T, jax.ops.index[:, b:], Tright)
    V = jax.ops.index_update(V, jax.ops.index[:, :b], jnp.dot(V[:, :b],
                             Vsmall_d))

    if verbose:
        print("*************")
        print("AV:", AV)
        print("AV1:", AV1, "AV2:", AV2)
        print("U:", U, "T11:", T11, "Vs:", Vsmall_d)
        print("Tright:", Tright)
        print("T:", T)
        print("V:", V)
        print("*************")
    output = [U, T, V]
    return output


def randUTV(A, b, q=1, p=0):
    """
    Performs the "optimized" randUTV in Figure4 of the paper.

    This is an interface function housing anything we don't want to Jit.

    Arguments
    ---------
    A: (m x n) matrix to be factorized.
    b (int): block size.
    q (int): Number of power iterations, a hyperparameter.
    p (int): Amount of oversampling, a hyperparameter. Currently does nothing.
    householder:If True, exploit the Householder structure of the QR
        decompositions to speed up the transformation.
    """
    Gwork = jnp.zeros((A.shape[0], b), dtype=A.dtype)
    U, T, V = __randUTV_work(A, Gwork, q=q, p=p)
    print("result:", U@T@dag(V))
    
    return [U, T, V]


def __randUTV_block(bj, b, B1, I2, J2, B3, B2B3, Gwork, U, T, V):
    print("block")
    # TODO: halt at desired accuracy.
    # Very strange things happen with Jit if we try to refactor
    # this code block into a subroutine!

    # Get array indexes ready.
    # print(B1)
    # print(I2)
    # print(J2)
    # print(B3)
    # print(B2B3)
    thisblock = T[B2B3, B2B3]
    ###################################################################
    # CODE TO EXPLOIT HOUSEHOLDER STRUCTURE BEGINS HERE
    ###################################################################
    Vh_W, Vh_YH, _ = rand_range_row_jit(thisblock, Gwork[bj:, :])
    T = index_update(T, index[:, B2B3],
                     qr.B_times_Q_WY(T[:, B2B3], Vh_W, Vh_YH))

    V = index_update(V, index[:, B2B3],
                     qr.B_times_Q_WY(V[:, B2B3], Vh_W, Vh_YH))

    # UH, R = jnp.linalg.qr(T[bi:, bi:J2end], mode="complete")
    Uh_W, Uh_YH, Uh_R = qr.house_qr(T[B2B3, J2], mode="WY")
    U = index_update(U, index[:, B2B3],
                     qr.B_times_Q_WY(U[:, B2B3], Uh_W, Uh_YH))
    T = index_update(T, index[B2B3, B3],
                     qr.Qdag_WY_times_B(T[B2B3, B3], Uh_W, Uh_YH))
    T = index_update(T, index[B3, J2], 0.)
    ###################################################################
    # CODE TO EXPLOIT HOUSEHOLDER STRUCTURE ENDS HERE
    ###################################################################

    Us, Ds, Vsh = jnp.linalg.svd(Uh_R[:(b-1), :(b-1)])
    Vs = dag(Vsh)
    T = index_update(T, index[I2, J2], jnp.diag(Ds))
    T = index_update(T, index[I2, B3], dag(Us)@T[I2, B3])
    U = index_update(U, index[:, I2], U[:, I2]@Us)
    T = cond(bj > 0,
             T, lambda x: index_update(x, index[B1, J2], x[B1, J2]@Vs),
             T, lambda x: x)

    # T = index_update(T, index[B1, J2], T[B1, J2]@Vs)
    V = index_update(V, index[:, J2], V[:, J2]@Vs)
    UTV = U@T@dag(V)
    print(UTV)
    return [U, T, V]


def __randUTV_final(bj, B1, B2B3, U, T, V):
    print("final")
    thisblock = T[B2B3, B2B3]
    Us, Dvals, Vsh = jnp.linalg.svd(thisblock, full_matrices=True)
    Vs = dag(Vsh)

    U = index_update(U, index[:, B2B3], U[:, B2B3]@Us)
    V = index_update(V, index[:, B2B3], V[:, B2B3]@Vs)

    idxs = matutils.subblock_main_diagonal(T, bi=bj)
    allDs = jnp.zeros(idxs[0].size)
    allDs = index_update(allDs, index[:Dvals.size], Dvals)
    T = index_update(T, index[B2B3, B2B3], 0.)
    T = index_update(T, idxs, allDs)
    T = cond(bj > 0,
             T, lambda x: index_update(x, index[B1, B2B3], x[B1, B2B3]@Vs),
             T, lambda x: x)
    #T = index_update(T, index[B1, B2B3], T[B1, B2B3]@Vs)
    return [U, T, V]

#@jax.jit
def __randUTV_work(A, Gwork, q=1, p=0):
    """
    Performs the "optimized" randUTV in Figure4 of the paper.

    Arguments
    ---------
    A: (m x n) matrix to be factorized.
    Gwork: (m x b) matrix that will be used as a work space for the
           randomized range finder.
    b (int): block size
    q (int): Number of power iterations, a hyperparameter.
    p (int): Amount of oversampling, a hyperparameter. Currently does nothing.
    householder:If True, exploit the Householder structure of the QR
        decompositions to speed up the transformation.
    """

    m, n = A.shape

    # Initialize output variables:
    T = A
    U = jnp.eye(m, dtype=A.dtype)
    V = jnp.eye(n, dtype=A.dtype)

    b = Gwork.shape[1]
    for bj in range(0, min(m, n), b):
        j = bj//b
        m, n = T.shape
        I2end = b*(j+1)-1
        J2end = b*(j+1)-1
        B1 = index[:(bj-1)]
        I2 = index[bj:I2end]
        J2 = index[bj:J2end]
        B3 = index[b*(j+1):]
        B2B3 = index[bj:]
        block_func = partial(__randUTV_block, bj, b, B1, I2, J2, B3, B2B3,
                             Gwork)
        final_func = partial(__randUTV_final, bj, B1, B2B3)
        if bj + b < m - 1 and bj + b < n - 1:
            U, T, V = block_func(U, T, V)
        else:
            U, T, V = final_func(U, T, V)
        # UTV = cond(
                    # (bj + b < m) and (bj + b < n),
                    # [U, T, V], lambda x: block_func(*x),
                    # [U, T, V], lambda x: final_func(*x)
                  # )
    return [U, T, V]
