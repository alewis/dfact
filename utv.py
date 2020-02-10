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


@partial(jax.jit, static_argnums=(1, 2, 3))
def rand_range_row_jit(A, b, q, p):
    """
    Jit implementation of
    randomized range finding. Starting from a general (m x n) matrix
    A, a (n x b) with b<=m matrix, Q, is sought such that
    ||I - Q Q^dag A|| is minimized.

    Q is returned in the "WY" form; that is, as matrices W and dag(Y) such that
    Q = I - W dag(Y).

    Arguments
    ---------
    A (array): The m x n matrix to be factorized.
    G (array): An m x b array that will be overwritten with random
               numbers.
    q (int)  : Number of power iterations
    p (int)  : Degree of oversampling.

    q and p are treated by Jit as static arguments.

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
    G = jnp.zeros((m, b+p), dtype=A.dtype)
    G = index_update(G, index[:], matutils.gaussian_random_fill(G))
    G0, _ = qr.house_qr(G, mode="reduced")

    Y = dag(A) @ G0
    Y0, _ = qr.house_qr(Y, mode="reduced")

    # Power iterations speed the decay of Y's singular values. This improves
    # the approximation, which is worse for smaller SVs.
    AdagA = dag(A) @ A
    AdagA0 = qr.house_qr(AdagA, mode="reduced")
    for _ in range(q):
        out_tup = qr.house_qr(AdagA@Y0, mode="reduced")
        Y0 = index_update(Y0, index[:], out_tup[0])

    Qout = qr.house_qr(Y0[:, :b], mode="WY")
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


def randUTV(A, b, q=2, p=0):
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
    U, T, V = __randUTV_work(A, b, q, p)
    return [U, T, V]


#  def __randUTV_block(bj, b, B1, I2, J2, B3, B2B3, Gwork, U, T, V):
#      #  print("***BLOCK***")
#      #  # TODO: halt at desired accuracy.
#      #  print("GOING IN")
#      #  print("U: \n", U)
#      #  print("T: \n", T)
#      #  print("V: \n", V)
#      #  print("UTV:\n ", U@T@dag(V))
#      thisblock = T[B2B3, B2B3]
#      ###################################################################
#      # CODE TO EXPLOIT HOUSEHOLDER STRUCTURE BEGINS HERE
#      ###################################################################
#      Vh_W, Vh_YH, _ = rand_range_row_jit(thisblock, Gwork[bj:, :])
#      T = index_update(T, index[:, B2B3],
#                       qr.B_times_Q_WY(T[:, B2B3], Vh_W, Vh_YH))

#      V = index_update(V, index[:, B2B3],
#                       qr.B_times_Q_WY(V[:, B2B3], Vh_W, Vh_YH))

#      # UH, R = jnp.linalg.qr(T[bi:, bi:J2end], mode="complete")
#      Uh_W, Uh_YH, Uh_R = qr.house_qr(T[B2B3, J2], mode="WY")
#      U = index_update(U, index[:, B2B3],
#                       qr.B_times_Q_WY(U[:, B2B3], Uh_W, Uh_YH))
#      T = index_update(T, index[B2B3, B3],
#                       qr.Qdag_WY_times_B(T[B2B3, B3], Uh_W, Uh_YH))
#      T = index_update(T, index[B3, J2], 0.)
#      ###################################################################
#      # CODE TO EXPLOIT HOUSEHOLDER STRUCTURE ENDS HERE
#      ###################################################################

#      Us, Ds, Vsh = jnp.linalg.svd(Uh_R[:b, :b])
#      Vs = dag(Vsh)
#      T = index_update(T, index[I2, J2], jnp.diag(Ds))
#      T = index_update(T, index[I2, B3], dag(Us)@T[I2, B3])
#      U = index_update(U, index[:, I2], U[:, I2]@Us)
#      T = index_update(T, index[B1, J2], T[B1, J2]@Vs)
#      #  T = cond(bj > 0,
#      #           T, lambda x: index_update(x, index[B1, J2], x[B1, J2]@Vs),
#      #           T, lambda x: x)

#      # T = index_update(T, index[B1, J2], T[B1, J2]@Vs)
#      V = index_update(V, index[:, J2], V[:, J2]@Vs)
#      #  print("GOING OUT")
#      #  print("U: \n", U)
#      #  print("T: \n", T)
#      #  print("V: \n", V)
#      #  print("UTV:\n ", U@T@dag(V))
#      return [U, T, V]

#  def __randUTV_final(bj, B1, B2B3, U, T, V):
#      #  print("***FINAL***")
#      #  print("GOING IN")
#      #  print("U: \n", U)
#      #  print("T: \n", T)
#      #  print("V: \n", V)
#      #  print("UTV:\n ", U@T@dag(V))
#      thisblock = T[B2B3, B2B3]

#      Us, Dvals, Vsh = jnp.linalg.svd(thisblock, full_matrices=True)
#      Vs = dag(Vsh)

#      U = index_update(U, index[:, B2B3], U[:, B2B3]@Us)
#      V = index_update(V, index[:, B2B3], V[:, B2B3]@Vs)

#      idxs = matutils.subblock_main_diagonal(T, bi=bj)
#      allDs = jnp.zeros(idxs[0].size)
#      allDs = index_update(allDs, index[:Dvals.size], Dvals)
#      T = index_update(T, index[B2B3, B2B3], 0.)
#      T = index_update(T, idxs, allDs)
#      T = index_update(T, index[B1, B2B3], T[B1, B2B3]@Vs)
#      #  T = cond(bj > 0,
#      #           T, lambda x: index_update(x, index[B1, B2B3], x[B1, B2B3]@Vs),
#      #           T, lambda x: x)
#      #T = index_update(T, index[B1, B2B3], T[B1, B2B3]@Vs)
#      #  print("GOING OUT")
#      #  print("U: \n", U)
#      #  print("T: \n", T)
#      #  print("V: \n", V)
#      #  print("UTV:\n ", U@T@dag(V))
#      return [U, T, V]

def divvy_blocks(bj, T, b):
    """
    This computes the active blocks for each loop of randUTV.
    """
    B1 = index[:bj]
    I2end = jnp.min([bj+b, T.shape[0]])
    J2end = jnp.min([bj+b, T.shape[1]])
    I2 = index[bj:I2end]
    J2 = index[bj:J2end]
    B3 = index[bj+b:]
    B2B3 = index[bj:]
    return [B1, I2, J2, B3, B2B3]


@partial(jax.jit, static_argnums=(1,))
def initialize_slices(T, b):
    B1s = []
    B2s = []
    B3s = []
    B2B3s = []
    bj0 = 0
    mindim = jnp.min(T.shape)

    for bj in range(0, mindim-b, b):
        bj0 = bj
        B1s.append(index[:bj])
        B2s.append(index[bj:bj+b])
        B3s.append(index[bj+b:])
        B2B3s.append(index[bj:])
    for bj in range(bj0+b, mindim, b):
        B1s.append(index[:bj])
        B2B3s.append(index[bj:])
    return [B1s, B2s, B3s, B2B3s]


@partial(jax.jit, static_argnums=(1, 2, 3))
def __randUTV_work(A, b, q, p):

    """
    Performs the "optimized" randUTV in Figure4 of the paper.

    Arguments
    ---------
    A: (m x n) matrix to be factorized.
    Gwork: (m x b) matrix that will be used as a work space for the
           randomized range finder.
    b (int): block size
    q (int): Number of power iterations, a hyperparameter.
    p (int): Amount of oversampling, a hyperparameter. 
    """

    m, n = A.shape

    # Initialize output variables:
    U = jnp.eye(m, dtype=A.dtype)
    T = A
    V = jnp.eye(n, dtype=A.dtype)

    B1s, B2s, B3s, B2B3s = initialize_slices(T, b)
    mindim = jnp.min(T.shape)
    bj0 = 0  # Passes final value to next for loop.
    for bj in range(0, mindim-b, b):
        bj0 = bj
        # During this for loop, we create and apply transformation matrices
        # bringing the j'th b x b diagonal block of T to diagonal form.
        # The loop terminates when the next diagonal block would either be
        # empty or smaller than b x b, in which case we execute the code
        # within the next for loop. We use a pair of for loops to avoid
        # the awkward interplay between conditionals and jit.
        j = bj//b
        B1, B2, B3, B2B3 = [B1s[j], B2s[j], B3s[j], B2B3s[j]]
        thisblock = T[B2B3, B2B3]

        # Use randomized sampling methods to generate a unitary matrix Vj
        # whose columns form an approximate orthonormal basis for those of
        # T([I2, J3], [J2, J3]); that is, the portion of A which is not
        # yet diagonal-ish. Vj is in its WY QR representation,
        # that is, as two matrices Vj_W and Vj_YH.
        Vj_W, Vj_YH, _ = rand_range_row_jit(thisblock, b, q, p)

        # Compute T = T @ Vj and V = V @ Vj using the function
        # qr.B_times_Q_WY, which does B @ Q with Q in the WY representation.
        # Since V is initially the identity, this builds up
        # V=V0@V0s@V1@V0s@V2... ,
        # so that V inherits the unitarity of its constituents. T@dag(V)
        # then reverses the procedure. V0s, which is also unitary,  is computed
        # in the final step of the for loop.
        T = index_update(T, index[:, B2B3],
                         qr.B_times_Q_WY(T[:, B2B3], Vj_W, Vj_YH))
        V = index_update(V, index[:, B2B3],
                         qr.B_times_Q_WY(V[:, B2B3], Vj_W, Vj_YH))

        # Build an orthonormal/unitary matrix Uj in similar fashion, and
        # compute U = U@Uj, T = dag(Uj)@T. Thus, U @ T again reverses the
        # procedure, while U remains unitary. Uj is also in its WY
        # representation. This time, we hang onto the matrix R in the QR
        # decomposition for later use.
        Uj_W, Uj_YH, Uj_R = qr.house_qr(T[B2B3, B2], mode="WY")
        U = index_update(U, index[:, B2B3],
                         qr.B_times_Q_WY(U[:, B2B3], Uj_W, Uj_YH))
        T = index_update(T, index[B2B3, B3],
                         qr.Qdag_WY_times_B(T[B2B3, B3], Uj_W, Uj_YH))
        # Zero out entries of T beneath the current block diagonal.
        T = index_update(T, index[B3, B2], 0.)

        # Uj_R[:b, :b] is now the portion of the active diagonal block which
        # we have not yet absorbed into U, T, or V. Diagonalize it with
        # an SVD to yield 'small' matrices Us@Ds@Vsh = svd(Uj_R[:b, :b].
        # T[I2, J2] = Ds thus diagonalizes the active block. Absorb
        # the unitary matrices Us and Vsh into U, T, and V so that the
        # transformation is reversed during A = U @ T @ dag(V).
        Us, Ds, Vsh = jnp.linalg.svd(Uj_R[:b, :b])
        Vs = dag(Vsh)
        T = index_update(T, index[B2, B2], jnp.diag(Ds))
        T = index_update(T, index[B2, B3], dag(Us)@T[B2, B3])
        U = index_update(U, index[:, B2], U[:, B2]@Us)
        T = index_update(T, index[B1, B2], T[B1, B2]@Vs)
        V = index_update(V, index[:, B2], V[:, B2]@Vs)

    for bj in range(bj0+b, mindim, b):
        # This 'loop' operates on the last diagonal block in the case that
        # b did not divide either m or n evenly. It performs the SVD
        # step at the end of the 'main' block, accomodating the relevant
        # matrix dimensions. This loop should only ever increment either
        # never or once and
        # would more naturally be an if statement, but Jit doesn't like that.
        B1 = B1s[-1]
        B2B3 = B2B3s[-1]
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
        T = index_update(T, index[B1, B2B3], T[B1, B2B3]@Vs)
    return [U, T, V]






