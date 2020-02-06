"""
Jax implementation of householder QR exposing the low-level functionality
needed for the UTV decomposition.

Adam GM Lewis
adam.lws@gmail.com
alewis@perimeterinstitute.ca
"""
import jax
from jax.ops import index_update, index_add, index
import jax.numpy as jnp
import numpy as np
import matutils
from matutils import dag

###############################################################################
# UTILITIES
###############################################################################
@jax.jit
def sign(x):
    """
    Sign function using the standard (?) convention sign(x) = x / |x| in
    the complex case. Returns 0 with the same type as x if x == 0.
    Note the numpy implementation uses the slightly different convention
    sign(x) = x / sqrt(x * x).
    """
    result = jax.lax.cond(x == 0,
                          x, lambda x: 0*x,  # 0 with the correct dtype if x==0
                          x, lambda x: x/jnp.abs(x))  # x/|x| otherwise
    return result


###############################################################################
# COMPUTATION OF HOUSEHOLDER VECTORS
###############################################################################
@jax.jit
def house(x):
    """
    Given a real or complex length-m vector x, finds the Householder vector
    v and its inverse normalization tau such that
                P = I - beta * v \otimes dag(v)
    is the symmetric (Hermitian) and orthogonal (unitary) Householder matrix
    representing reflections about x.

    Returns a list [v, beta], where v is a length-m vector whose first
    component is 1, and beta = 2/(dag(v) v).

    x will be treated as a flattened vector whatever its shape.

    Parameters
    ----------
    x:  array_like, shape(m,)
        The vector about which to compute the Householder reflector. Will
        be flattened (inside this fucntion only) to the prescribed shape.

    Output
    ------
    [v, beta]:
        v: array_like, shape(m,), the Householder vector including the 1.
        beta: float, the normalization 2/|v|
    """
    x = x.ravel()
    x_2_norm = jnp.linalg.norm(x[1:])
    # The next two lines are logically equivalent to
    # if x_2_norm == 0:
    #   v, beta = __house_zero_norm(x)
    # else:
    #   v, beta = __house_nonzero_norm( (x, x_2_norm) )
    switch = (x_2_norm == 0)
    v, beta = jax.lax.cond(switch,
                           x, __house_zero_norm,
                           (x, x_2_norm), __house_nonzero_norm)
    return [v, beta]


@jax.jit
def __house_zero_norm(x):
    """
    Handles house(x) in the case that norm(x[1:])==0.
    """
    beta = 2.
    v = x
    v = index_update(v, index[0], 1.)
    beta = jax.lax.cond(x[0] == 0,
                        x, lambda x: x[0]*0,
                        x, lambda x: x[0]*0 + 2
                        ).real
    return [v, beta]


@jax.jit
def __house_nonzero_norm(xtup):
    """
    Handles house(x) in the case that norm(x[1:])!=0.
    """
    x, x_2_norm = xtup
    x, x_2_norm = xtup
    x_norm = jnp.linalg.norm(jnp.array([x[0], x_2_norm]))
    rho = sign(x[0])*x_norm

    v_1p = x[0] + rho
    v_1pabs = jnp.abs(v_1p)
    v_1m = x[0] - rho
    v_1mabs = jnp.abs(v_1m)

    # Pick whichever of v[0] = x[0] +- sign(x[0])*||x||
    # has greater ||v[0]||, and thus leads to greater ||v||.
    # Golub and van Loan prescribes this "for stability".
    v_1, v_1abs = jax.lax.cond(v_1pabs >= v_1mabs,
                               (v_1p, v_1pabs), lambda x: x,
                               (v_1m, v_1mabs), lambda x: x)

    v = x
    v = index_update(v, index[1:], v[1:]/v_1)
    v = index_update(v, index[0], 1.)
    v_2_norm = x_2_norm / v_1abs
    v_norm_sqr = 1 + v_2_norm**2
    beta = (2 / v_norm_sqr).real
    return [v, beta]


###############################################################################
# MANIPULATION OF HOUSEHOLDER VECTORS
###############################################################################
@jax.jit
def form_dense_P(hlist):
    """
    Computes the dense Householder matrix P = I - beta * (v otimes dag(v))
    from the Householder reflector stored as hlist = (v, beta). This is
    useful for testing.
    """
    v, beta = hlist
    Id = jnp.eye(v.size, dtype=v.dtype)
    P = Id - beta * jnp.outer(v, dag(v))
    return P


@jax.jit
def house_leftmult(A, v, beta):
    """
    Given the m x n matrix A and the length-n vector v with normalization
    beta such that P = I - beta v otimes dag(v) is the Householder matrix that
    reflects about v, compute PA.

    Parameters
    ----------
    A:  array_like, shape(M, N)
        Matrix to be multiplied by H.

    v:  array_like, shape(N).
        Householder vector.

    beta: float
        Householder normalization.

    Returns
    -------
    C = PA
    """
    C = A - jnp.outer(beta*v, jnp.dot(dag(v), A))
    return C


@jax.jit
def house_rightmult(A, v, beta):
    """
    Given the m x n matrix A and the length-n vector v with normalization
    beta such that P = I - beta v otimes dag(v) is the Householder matrix that
    reflects about v, compute AP.

    Parameters
    ----------
    A:  array_like, shape(M, N)
        Matrix to be multiplied by H.

    v:  array_like, shape(N).
        Householder vector.

    beta: float
        Householder normalization.

    Returns
    -------
    C = AP
    """
    C = A - jnp.outer(A@v, beta*dag(v))
    return C


###############################################################################
# MANIPULATION OF FACTORED QR REPRESENTATION
###############################################################################
def factored_rightmult_dense(A, H, betas):
    """
    Computes C = A * Q, where Q is in the factored representation.
    With A = Hbetalist[0].shape[0], this computes Q, but less economically
    than 'factored_to_QR'.

    This is a dense implementation written to test 'factored_rightmult' below.
    Do not call it in production code.
    """
    C = A
    n = C.shape[1]
    for j, beta in enumerate(betas):
        vnz = jnp.array([1.]+list(H[j+1:, j]))
        nminus = n - vnz.size
        v = jnp.array([0.]*nminus + list(vnz))
        P = form_dense_P([v, beta])
        C = index_update(C, index[:, :], C@P)
    return C


@jax.jit
def factored_rightmult(A, H, betas):
    """
    Computes C = A * Q, where Q is in the factored representation.
    With A = Hbetalist[0].shape[0], this computes Q, but less economically
    than 'factored_to_QR'.
    """
    C = A
    for j, beta in enumerate(betas):
        v = jnp.array([1.]+list(H[j+1:, j]))
        C = index_update(C, index[:, j:], house_rightmult(C[:, j:], v, beta))
    return C


@jax.jit
def factored_to_QR(h, beta):
    """
    Computes dense matrices Q and R from the factored QR representation
    [h, tau] as computed by qr with mode == "factored".
    """
    m, n = h.shape
    R = jnp.triu(h)
    Q = jnp.eye(m, dtype=h.dtype)
    for j in range(n-1, -1, -1):
        v = jnp.concatenate((jnp.array([1.]), h[j+1:, j]))
        Q = index_update(Q, index[j:, j:],
                         house_leftmult(Q[j:, j:], v, beta[j]))
    out = [Q, R]
    return out

###############################################################################
# MANIPULATION OF WY QR REPRESENTATION
###############################################################################
@jax.jit
def times_householder_vector(A, H, j):
    """
    Computes A * v_j where v_j is the j'th Householder vector in H.

    Parameters
    ----------
    A: k x M matrix to multiply by v_j.
    H: M x k matrix of Householder reflectors.
    j: The column of H from which to extract v_j.

    Returns
    ------
    vout: length-M vector of output.
    """

    vin = jnp.array([1.]+list(H[j+1:, j]))
    vout = jnp.zeros(H.shape[0], dtype=H.dtype)
    vout = index_update(vout, index[j:], A[:, j:] @ vin)
    return vout


@jax.jit
def factored_to_WY(hbetalist):
    """
    Converts the 'factored' QR representation [H, beta] into the WY
    representation, Q = I - WY^H.

    Parameters
    ----------
    hbetalist = [H, beta] : list of array_like, shapes [M, N] and [N].
        'factored' QR rep of a matrix A (the output from
        house_QR(A, mode='factored')).

    Returns
    -------
    [W, YH]: list of ndarrays of shapes [M, N].
        The matrices W and Y generating Q along with R in the 'WY'
        representation.
    -W (M x N): The matrix W.
    -YH (M x N): -Y is the lower triangular matrix with the essential parts of
                  the Householder reflectors as its columns,
                  obtained by setting the main diagonal of H to 1 and zeroing
                  out everything above.
                 -YH, the h.c. of this matrix, is thus upper triangular
                  with the full Householder reflectors as its rows. This
                  function returns YH, which is what one needs to compute
                  C = Q @ B = (I - WY^H) @ B = B - W @ Y^H @ B.

                  Note: it is more efficient to store W and Y^H separately
                        than to precompute their product, since we will
                        typically have N << M when exploiting this
                        representation.
    """

    H, betas = hbetalist
    m, n = matutils.matshape(H)
    W = jnp.zeros(H.shape, H.dtype)
    vj = jnp.array([1.]+list(H[1:, 0]))
    W = index_update(W, index[:, 0], betas[0] * vj)

    Y = jnp.zeros(H.shape, H.dtype)
    Y = index_update(Y, index[:, 0], vj)
    for j in range(1, n):
        vj = index_update(vj, index[j+1:], H[j+1:, j])
        vj = index_update(vj, index[j], 1.)  # vj[j:] stores the current vector
        YHv = (dag(Y[j:, :j])) @ vj[j:]
        z = W[:, :j] @ YHv
        z = index_add(z, index[j:], -vj[j:])
        z = index_update(z, index[:], -betas[j]*z)

        W = index_update(W, index[:, j], z)
        Y = index_update(Y, index[j:, j], vj[j:])
    YH = dag(Y)
    return [W, YH]


@jax.jit
def B_times_Q_WY(B, W, YH):
    """
    Computes C(kxm) = B(kxm)@Q(mxm) with Q given as W and Y^H in
    Q = I(mxm) - W(mxr)Y^T(rxm).
    """
    C = B - (B@W)@YH
    return C

@jax.jit
def Qdag_WY_times_B(B, W, YH):
    """
    Computes C(mxk) = QH(mxm)@B(mxk) with Q given as W and Y^H in
    Q = I(mxm) - W(mxr)Y^T(rxm)
    """

@jax.jit
def WY_to_Q(W, YH):
    """
    Retrieves Q from its WY representation.
    """
    m = W.shape[0]
    Id = jnp.eye(m, dtype=W.dtype)
    return B_times_Q_WY(Id, W, YH)


###############################################################################
# QR DECOMPOSITION
###############################################################################
def house_qr(A, mode="reduced"):
    """
    Performs a QR decomposition of the m x n real or complex matrix A
    using the Householder algorithm.

    The string parameter 'mode' determines the representation of the output.
    In this way, one can retrieve various implicit representations of the
    factored matrices. This can be a significant optimization in the case
    of a highly rectangular A, which is the reason for this function's
    existence.

    Parameters
    ----------
    A : array_like, shape (M, N)
            Matrix to be factored.

        mode: {'reduced', 'complete', 'r', 'factored', 'WY'}, optional
            If K = min(M, N), then:
              - 'reduced': returns Q, R with dimensions (M, K), (K, N)
                (default)
              - 'complete': returns Q, R  with dimensions (M, M), (M, N)
              - 'r': returns r only with dimensions (K, N)
              - 'factored': returns H, beta with dimensions (N, M), (K), read
                 below for details.
              - 'WY' : returns W, Y with dimensions (M, K), read below for
                 details.

    With 'reduced', 'complete', or 'r', this function simply passes to
    jnp.linalg.qr, which depending on the currect status of Jax may lead to
    NotImplemented if A is complex.

    With 'factored' this function returns the same H, beta as generated by
    the Lapack function dgeqrf() (but in row-major form). Thus,
    H contains the upper triangular matrix R in its upper triangle, and
    the j'th Householder reflector forming Q in the j'th column of its
    lower triangle. beta[j] contains the normalization factor of the j'th
    reflector, called 'beta' in the function 'house' in this module.

    The matrix Q is then represented implicitly as
        Q = H(0) H(1) ... H(K), H(i) = I - tau[i] v dag(v)
    with v[:j] = 0; v[j]=1; v[j+1:]=A[j+1:, j].

    Application of Q (C -> dag{Q} C) can be made directly from this implicit
    representation using the function factored_multiply(C). When
    K << max(M, N), both the QR factorization and multiplication by Q
    using factored_multiply theoretically require far fewer operations than
    would an explicit representation of Q. However, these applications
    are mostly Level-2 BLAS operations.

    With 'WY' this function returns (M, K) matrices W and Y such that
        Q = I - W dag(Y).
    Y is lower-triangular matrix of Householder vectors, e.g. the lower
    triangle
    of the matrix H resulting from mode='factored'. W is then computed so
    that the above identity holds.

    Application of Q can be made directly from the WY representation
    using the function WY_multiply(C). The WY representation is
    a bit more expensive to compute than the factored one, though still less
    expensive than the full Q. Its advantage versus 'factored' is that
    WY_multiply calls depend mostly on Level-3 BLAS operations.


    Returns
    -------
    Q: ndarray of float or complex, optional
        The column-orthonormal orthogonal/unitary matrix Q.

    R: ndarray of float or complex, optional.
        The upper-triangular matrix.

    [H, beta]: list of ndarrays of float or complex, optional.
        The matrix H and scaling factors beta generating Q along with R in the
        'factored' representation.

    [W, Y, R] : list of ndarrays of float or complex, optional.
        The matrices W and Y generating Q along with R in the 'WY'
        representation.

    Raises
    ------
    LinAlgError
        If factoring fails.

    NotImplementedError
        In reduced, complete, or r mode with complex ijnp.t.
        In factored or WY mode in the case M < N.
    """
    if mode == "reduced" or mode == "complete" or mode == "r":
        return jnp.linalg.qr(A, mode=mode)
    else:
        m, n = A.shape
        if n > m:
            raise NotImplementedError("n > m QR not implemented in factored"
                                      + "or WY mode.")
        if mode == "factored":
            return __house_qr_factored(A)
        elif mode == "WY":
            hbetalist = __house_qr_factored(A)
            R = jnp.triu(hbetalist[0])
            WYlist = factored_to_WY(hbetalist)
            output = WYlist + [R]
            return output
        else:
            raise ValueError("Invalid mode: ", mode)


@jax.jit
def __house_qr_factored(A):
    """
    Computes the QR decomposition of A in the 'factored' representation.
    This is a workhorse function to be accessed externally by
    house_qr(A, mode="factored"), and is documented more extensively in
    that function's documentation.

    """
    H = A
    M, N = matutils.matshape(A)
    beta = list()
    for j in range(A.shape[1]):
        v, thisbeta = house(H[j:, j])
        beta.append(thisbeta)
        H = index_update(H, index[j:, j:], house_leftmult(H[j:, j:], v,
                                                          thisbeta))
        if j < M:
            H = index_update(H, index[j+1:, j], v[1:])
    beta = jnp.array(beta)
    output = [H, beta]
    return output








