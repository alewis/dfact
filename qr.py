"""
Jax implementation of householder QR exposing the low-level functionality
needed for the UTV decomposition.

Adam GM Lewis
adam.lws@gmail.com
alewis@perimeterinstitute.ca
"""
import jax
from jax.ops import index_update, index
import jax.numpy as jnp
import numpy as np
import matutils
from matutils import dag

@jax.jit
def sign(x):
    # if x == 0:
    #   return 0
    # else:
    #   return x / jnp.abs(x)
    return jax.lax.cond(x == 0, x, lambda x: 0*x, x, lambda x: x/jnp.abs(x))

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
    # x = x.ravel()
    
    # print("x:",x)
    # rho = sign(x[0]) * jnp.linalg.norm(x)
    # v1 = x[0] - rho
    # v2 = x[1:] / v1
    # beta = 2 / (1 + jnp.dot(dag(v2), v2))
    # v = jnp.array([1.] + list(v2))
    # v0 = jnp.norm(x) * sign(x[0])
    # v = jnp.array([1.] + x[1:]/v0).astype(x.dtype)
    # sigma = jnp.linalg.norm(x[1:])
    # beta = 2*(v0**2)
    #beta = 2 / jnp.dot(v, dag(v))

    x = x.ravel()
    x_2_norm = jnp.linalg.norm(x[1:])
    # The next two lines are logically equivalent to
    # if x_2_norm == 0:
       # v, beta = __house_zero_norm(x)
    # else:
       # v, beta = __house_nonzero_norm( (x, x_2_norm) )
    switch = (x_2_norm == 0)
    v, beta = jax.lax.cond(switch, x, __house_zero_norm, (x, x_2_norm),
                           __house_nonzero_norm)
    return [v, beta]

@jax.jit
def __house_zero_norm(x):
    """
    Handles house(x) in the case that norm(x[1:])==0.
    """
    beta = 2.
    v = jnp.array([1.] + list(x[1:]), dtype=x.dtype)
    #beta = jax.lax.cond(x[0]<0, 1., lambda x: 0., 1, lambda x: 0.) 
    return [v, beta]


@jax.jit
def __house_nonzero_norm(xtup):
    """
    Handles house(x) in the case that norm(x[1:])!=0.
    """
    x, x_2_norm = xtup
    # v_1 = x[0] - jnp.linalg.norm(x)#+ x_2_norm / jnp.abs(x[0])
    # vraw = jnp.array([v_1]+list(x[1:]), dtype=x.dtype)
    # v = vraw / jnp.linalg.norm(vraw)
    # beta = 2/(jnp.vdot(v, v))
    # beta = beta.real
    # print(beta)
    # v_2 = x[1:] / v_1
    # v_2_norm = jnp.linalg.norm(v_2) 
    # vraw = jnp.array([v_1]+list(v_2), dtype=x.dtype)
    # v = vraw / jnp.linalg.norm(v)
    #v0 = x[0] + sign(x[0])*jnp.linalg.norm(x)
    
    x, x_2_norm = xtup
    x_norm = jnp.linalg.norm(jnp.array([x[0], x_2_norm]))
    v_1 = x[0] + sign(x[0])*jnp.linalg.norm(x)
    v_2 = x[1:] / v_1
    v = jnp.array([1.]+list(v_2), dtype=x.dtype)
    beta = (2 / (v@dag(v))).real
    


    # x, x_2_norm = xtup
    # x_norm = jnp.linalg.norm(jnp.array([x[0], x_2_norm]))
    #v_1 = x[0] - x_norm

    # v_2 = x[1:] / v_1
    # v_2_norm = x_2_norm / jnp.abs(v_1)
    # v_norm_sqr = 1 + v_2_norm**2
    # beta = 2 / v_norm_sqr
    #v = jnp.array([1.]+list(v_2), dtype=x.dtype)
    return [v, beta]


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


def factored_rightmult_dense(A, H, betas):
    """
    Computes C = A * Q, where Q is in the factored representation.
    With A = Hbetalist[0].shape[0], this computes Q, but less economically
    than 'factored_to_QR'.
    """
    C = A
    n = C.shape[1]

    for j, beta in enumerate(betas):
        vnz = jnp.array([1.]+list(H[j+1:, j]))
        nminus = n - vnz.size
        v = jnp.array([0.]*nminus + list(vnz))
        P = form_dense_P([v, beta])
        C = index_update(C, index[:,:], C@P)
        #C = index_update(C, index[:, j:], house_rightmult(C[:, j:], v, beta))
    return C

def factored_rightmult(A, H, betas):
    """
    Computes C = A * Q, where Q is in the factored representation.
    With A = Hbetalist[0].shape[0], this computes Q, but less economically
    than 'factored_to_QR'.
    """
    C = A
    n = C.shape[1]

    for j, beta in enumerate(betas):
        v = jnp.array([1.]+list(H[j+1:, j]))
        #nminus = n - vnz.size
        #v = jnp.array([0.]*nminus + list(vnz))
        C = index_update(C, index[:, j:], house_rightmult(C[:, j:], v, beta))
        #C2 = index_update(C, index[:, j:], house_rightmult(C[:, j:], v, beta))
        #print(C, C2)
        
    return C

# def factored_mult_slow(A, H, betas):
    # C= = A
    # for j, beta in enumerate(betas):
        # v = 

def factored_multiply(hbetalist, C):
    """
    Does O = dag(Q) C with Q in the factored representation.
    """
    H, beta = hbetalist
    newC = C
    m, n = matutils.matsize(H)
    for j in range(n):
        thiscol = m - (j + 1)
        v = jnp.ones(thiscol)
        v = index_update(v, index[1:], C[j+1:, j])
        factor = -beta[j] * jnp.outer(v, dag(v)@C[j:, :])
        newC = index_update(C, index[j:, :], factor)
    return newC


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

    [W, Y] : list of ndarrays of float or complex, optional.
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
    elif mode == "factored":
        return __house_qr_factored(A)
    elif mode == "WY":
        hbetalist = __house_qr_factored(A)
        WYlist = factored_to_WY(hbetalist)
        return WYlist


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


def factored_to_WY(hbetalist):
    """
    Converts the 'factored' QR representation [H, beta] into the WY
    representation WY.

    Parameters ----------
    hbetalist = [H, beta] : list of array_like, shapes [M, N] and [N].
        'factored' QR rep of a matrix A (the output from
        house_QR(A, mode='factored')).

    Returns
    -------
    [W, Y]: list of ndarrays of shapes [M, N].
        The matrices W and Y generating Q along with R in the 'WY'
        representation.

    """

    # TODO: Investigate whether it would be more efficient to return dag(W).
    H, beta = hbetalist
    Y = jnp.tril(H)

    W = jnp.zeros(H.shape, H.dtype)
    W = index_update(W, index[:, 0], beta[0] * H[:, 0])
    m, r = matutils.matshape(H)
    for j in range(1, r):
        vj = H[j:, j]
        zf = beta[j] * (jnp.ones(m, dtype=H.dtype) - jnp.dot(W, dag(Y)))
        z = jnp.dot(zf, vj)
        W = index_update(W, index[:, j], z)
    return [W, Y]




def WY_multiply(WYlist, C):
    """
    Does O = dag(Q) C with Q in the WY representation.
    """
    W, Y = WYlist
    WC = jnp.dot(dag(W), C)
    out = C - jnp.dot(Y, WC)
    return out






