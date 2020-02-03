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
import jax.numpy as jnp
import numpy as np
import matutils
from matutils import dag
import math


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
    try:
        m, n = A.shape
    except ValueError:
        raise ValueError("A had invalid shape: ", A.shape)

    if b is None or b > n:
        b = n

    assert b > 0
    assert n_iter >= 0

    # G = jnp.random.randn(m, b)  # The Gaussian random subspace.
    G = matutils.gaussian_random(shape=(m, b), dtype=A.dtype)
    Y = jnp.dot(dag(A), G)  # (n x b) projection of A onto that subspace.

    if n_iter > 0:  # Power iterations speed the decay of Y's singular values.
        AdagA = jnp.dot(dag(A), A)
        for _ in range(n_iter):
            Y = jnp.dot(AdagA, Y)

    Q, _ = jnp.linalg.qr(Y, mode=mode)
    # Now Q is unitary with the same column space as Y.
    # Q is a (n x b) or (nxn) matrix.
    return Q


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

    assert m >= n
    assert b > 0
    assert p >= 0
    assert n_iter >= 0

    V = rand_range_row(A, b=b+p, n_iter=n_iter, mode="complete")  # (n x n)

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


def randUTV(A, b=None, q=1, p=0, householder=False):
    """
    Performs the "optimized" randUTV in Figure4 of the paper.

    Arguments
    ---------
    A: (m x n) matrix to be factorized.
    b (int): block size
    q (int): Number of power iterations, a hyperparameter.
    p (int): Amount of oversampling, a hyperparameter. Currently does nothing.
    householder:If True, exploit the Householder structure of the QR
        decompositions to speed up the transformation.
    """
    m, n = matutils.matshape(A)
    if b is None:
        b = n

    # Initialize output variables:
    T = A
    U = jnp.eye(m, dtype=A.dtype)
    V = jnp.eye(n, dtype=A.dtype)
    for i in range(min(math.ceil(m/b), math.ceil(n/b))):
        # TODO: halting criterion.
        # Get array indexes ready.
        bi = b*i
        I2end = min(b*(i+1), m)
        J2end = min(b*(i+1), n)
        thisblock = T[bi:, bi:]

        if (I2end < m and J2end < n):  # I3 and J3 are both nonempty.

            ###################################################################
            # CODE TO EXPLOIT HOUSEHOLDER STRUCTURE BEGINS HERE
            ###################################################################
            VH = rand_range_row(thisblock, b=b, n_iter=q, mode="complete")
            T = index_update(T, index[:, bi:], jnp.dot(T[:, bi:], VH))
            V = index_update(V, index[:, bi:], jnp.dot(V[:, bi:], VH))

            UH, R = jnp.linalg.qr(T[bi:, bi:J2end], mode="complete")
            U = index_update(U, index[:, bi:], jnp.dot(U[:, bi:], UH))
            T = index_update(T, index[bi:, J2end:],
                             jnp.dot(UH, T[bi:, J2end:]))
            ###################################################################
            # CODE TO EXPLOIT HOUSEHOLDER STRUCTURE ENDS HERE
            ###################################################################

            T = index_update(T, index[I2end:, bi:], 0.)

            Us, Ds, Vsh = jnp.linalg.svd(R[:b, :b])
            Vs = dag(Vsh)
            T = index_update(T, index[bi:I2end, bi:J2end], jnp.diag(Ds))
            T = index_update(T, index[bi:I2end, J2end:], jnp.dot(dag(Us),
                             T[bi:I2end, J2end:]))
            U = index_update(U, index[:, bi:I2end], jnp.dot(U[:, bi:I2end],
                             Us))
            T = index_update(T, index[:bi, bi:J2end],
                             jnp.dot(T[:bi, bi:J2end], Vs))
            V = index_update(V, index[:, bi:J2end], jnp.dot(V[:, bi:J2end],
                             Vs))

        else:  # One of I3 and J3 are empty, so this is the final block.
            Us, Dvals, Vsh = jnp.linalg.svd(thisblock, full_matrices=True)
            Vs = dag(Vsh)

            U = index_update(U, index[:, bi:], jnp.dot(U[:, bi:], Us))
            V = index_update(V, index[:, bi:], jnp.dot(V[:, bi:], Vs))

            idxs = matutils.subblock_main_diagonal(T, bi=bi)
            allDs = jnp.zeros(idxs[0].size)
            allDs = index_update(allDs, index[:Dvals.size], Dvals)
            T = index_update(T, index[bi:, bi:], 0.)
            T = index_update(T, idxs, allDs)
            T = index_update(T, index[:bi, bi:],
                             jnp.dot(T[:bi, bi:], Vs))

    output = [U, T, V]
    return output
