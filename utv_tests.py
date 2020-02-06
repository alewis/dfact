"""
Tests for functions in utv.py and qr.py.
To run all tests from the command line: $ python utv_tests.py

To run the tests instantiated in 'def suite()' near the end of this file
from a Colab or Jupyter Notebook:
    *in the notebook*
    import utv_tests
    import unittest

    def run_tests():
        suite = utv_tests.suite()
        runner = unittest.TextTestRunner()
        runner.run(suite)
    run_tests()

To add the tests in a subclass to 'def suite()', so that they will also
be run from the Colab or Jupyter notebook:
    *in this file*
    class NewTestSubClass(unittest.TestCase):

    def suite():
        suite = unittest.TestSuite()
        *one of the below for each subclass*
        suite.addTests(unittest.makeSuite(TestSubclass, 'test'))

A base class GaussianMatrixTest(unittest.TestCase) is provided to
loop over Gaussian matrices, treating each of several permutations of shape and
dtype as a different subtest. See that class' docstring for details.


Tests are currently defined that mean to do each of the following:
    TODO: FILL THESE IN
    test_replace_diagonal: Checks that matutils.replace_diagonal(A, D)
                           correctly returns a matrix identical to A, with
                           first D.size entries replaced by those of D, and
                           the rest zeroed out.
"""

import jax.numpy as jnp
import jax
import numpy as np
import unittest
import utv
import qr
import matutils
from matutils import dag
import itertools
import math


###############################################################################
# BASE CLASSES AND UTILITIES
###############################################################################
def errstring(arr1, name1, arr2, name2):
    """
    Utility function to generate failure messages in a few tests. Returns
    the Frobenius norm of the difference between arr1 and arr2, and a
    string errstring to be output if desired.
    """
    error = matutils.frob(arr1, arr2)
    errormsg = "Error = " + str(error) + "\n"
    errormsg += name1 + ":\n " + str(arr1) + "\n"
    errormsg += name2 + ":\n " + str(arr2)
    return (error, errormsg)


def errorstack(errtups, passed=True, msg="\n", thresh=1E-6):
    """
    Combines the output from multiples calls to errstring into a single
    pass-or-fail condition, based on comparison of each errtups[i][0] to
    thresh. Returns this flag along with a single error message, concatenated
    from
    those of each individual call.
    """
    for error, errormsg in errtups:
        if error < thresh:
            passed = False
            msg += "**********************************************************"
            msg += errormsg
            msg += "**********************************************************"
    return (passed, msg)


def manyclose(pairs):
    """
    Loops over pairs of numpy arrays and runs allclose on each. Returns True
    if allclose passes for each pair and False otherwise.
    """
    passed = True
    for pair in pairs:
        passed = passed and jnp.allclose(pair[0], pair[1])
    return passed


class GaussianMatrixTest(unittest.TestCase):
    """
    Concrete base class providing a set of random m x n matrices of differing
    data types to operate on.

    The parameter lists (ms, ns, dtypes) defined in the body of __init__
    will
    be used to generate a shape (m, n), dtype=dtype matrix, as if from
    nested for loops. Each will define a new subtest. The parameters to be
    looped over are fixed for any particular subclass of GaussianMatrixTest.
    Each subclass defines its choice of parameters by specializing
    __init__.

    Subclasses of GaussianMatrixTest should thus employ the following maneuvers
    to define tests:

    class MySubClass(GaussianMatrixTest):
        def __init__(self, *args, **kwargs):
            *specialize this class only if you want to change the default*
            *parameters*
            self.ns = (*start, stop, step*)
            self.ms = (*start, stop, step*)
            self.dtypes = [jnp.float32, jnp.complex64...]
            super().__init__(*args, **kwargs)


        def test_something(self, **kwargs):
            def impl(A, paramtup):
                m, n, dtype = paramtup
                ***body of test acting on the random input matrix A***
            self.iterloop(impl)


    *in def suite(): defined near the end of this file)*
    def suite():
        suite.addTests(unittest.makeSuite(MySubClass, 'test')

    *in a colab notebook from which you wish to run tests*

    """
    def __init__(self, *args, ns=(1, 6, 2),
                 ms=(6, 18, 6),
                 dtypes=[jnp.float32, jnp.complex64],
                 **kwargs):
        self.ns = range(*ns)
        self.ms = range(*ms)
        self.dtypes = dtypes
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.matrices = [matutils.gaussian_random(shape=(m, n), dtype=dtype)
                         for m, n, dtype
                         in itertools.product(self.ms, self.ns, self.dtypes)]

    def iterloop(self, func):
        """
        Iterates over the parameters defined in setUpImpl, stores them in
        paramtup, generates a random
        matrix A for each, and calls func(A, paramtup) as a new subtest.
        func(A, paramtup) should thus store the body of the test, and is
        usually defined within class methods as 'impl'.
        iterloop would most naturally be a decorator, but my Python
        isn't up to getting the interpreter to treat a class method as such.
        """
        params = itertools.product(self.ms, self.ns, self.dtypes)
        for A, paramtup in zip(self.matrices, params):
            m, n, dtype = paramtup
            with self.subTest(m=m, n=n, dtype=dtype):
                func(A, paramtup)


###############################################################################
# QR DECOMPOSITION TESTS
###############################################################################
class ExplicitQRTests(unittest.TestCase):
    """
    These tests check whether our QR code functions correctly, by creating
    a matrix with explicitly known input (self.testA) and ensuring we
    retrieve explicitly known results.
    """
    def setUp(self):
        self.testA = jnp.array([[1., -4.],
                                [2., 3.],
                                [2., 2.]])

    def test_householder_generation_on_explicit_input(self):
        """
        Checks that qr.house gives the correct output for known input.
        With x = [1, 2, 2] we should have v = [1, -1, -1]^T and
        beta = 2/3.

        ***THIS TEST SEEMS TO BE WRONG***
        """
        test_me = self.testA[:, 0]
        v, beta = qr.house(test_me)
        success = jnp.allclose(jnp.array([beta]), jnp.array([2./3.]))
        self.assertTrue(success, msg="beta[0] = "+str(beta)+" was wrong.")
        success = jnp.allclose(v, jnp.array([1., -1., -1.]))
        self.assertTrue(success, msg="v[0] = "+str(v)+" was wrong.")

    def test_factored_qr_on_explicit_input(self):
        """
        Checks that qr.house_qr(mode="factored") gives the correct output.

        ***THIS TEST SEEMS TO BE WRONG***
        """
        test_me = self.testA
        # print("\n")
        # print("*****************************")
        H, beta = qr.house_qr(test_me, mode="factored")
        print("H:\n ", H)
        print("beta: ", beta)
        Hnp, betanp = np.linalg.qr(test_me, mode="raw")
        print("Hnp:\n ", Hnp)
        print("betanp: ", betanp)
        correct_beta = jnp.array([2./3., 8./5.])
        # print("beta:", beta)
        # print("1/beta:", 1/beta)
        # print("correct_beta:", correct_beta)
        # print("*****************************")
        self.assertTrue(jnp.allclose(beta, correct_beta),
                        msg="beta="+str(beta)+" was wrong.")
        correct_H = jnp.array([[3., 2.],
                               [-1., 5.],
                               [-1., 0.5]])
        self.assertTrue(jnp.allclose(H, correct_H),
                        msg="\nH=\n"+str(H)+" was wrong.")

        correct_Q = jnp.array([[1./3, -14./15, -2./15],
                               [2./3, 1./3, -2./3],
                               [2./3, 2./15, 11./15]
                               ])
        Q, R = qr.factored_to_QR(H, beta)
        self.assertTrue(jnp.allclose(Q, correct_Q),
                        msg="\nQ=\n"+str(Q)+" was wrong.")

        QR = jnp.dot(Q, R)
        self.assertTrue(jnp.allclose(QR, test_me),
                        msg="\nQR=\n"+str(QR)+" was wrong.")


class TestHouseholderVectorProperties(GaussianMatrixTest):
    """
    Tests the code to compute and apply Householder reflections.
    """
    def __init__(self, *args, **kwargs):
        ns = (1, 2, 1)
        ms = (1, 5, 1)
        super().__init__(*args, ns=ns, ms=ms, **kwargs)

    def test_householder_unitarity(self, thresh=1E-6):
        """
        Random (m,) vectors are generated, and Householder reflections
        (v, beta) computed from them. The dense matrix
        P = I_m - beta v otimes dag(v) is formed, and its unitarity
        (orthogonality) is confirmed.
        """
        def impl(A, paramtup):
            v, beta = qr.house(A)
            P = qr.form_dense_P([v, beta])
            Pd = dag(P)
            unitary1 = jnp.dot(P, Pd)
            unitary2 = jnp.dot(Pd, P)
            Id = jnp.eye(v.size, dtype=P.dtype)
            err1, errormsg1 = errstring(unitary1, "P Pd", Id, "I")
            err2, errormsg2 = errstring(unitary2, "Pd P", Id, "I")
            errormsg = ""
            passed = True
            if err1 > thresh:
                passed = False
                errormsg += "\n" + errormsg1
            if err2 > thresh:
                passed = False
                errormsg += "\n" + errormsg2
            self.assertTrue(passed, msg=errormsg)
        self.iterloop(impl)

    # def test_householder(self, thresh=1E-6):
        # """
        # Random (m,) vectors are generated, and Householder reflections
        # (v, beta) computed from them. The dense matrix
        # P = I_m - beta v otimes dag(v) is formed, and its unitarity
        # (orthogonality) is confirmed.
        # """
        # def impl(A, paramtup):
            # v, beta = qr.house(A)

            # x0 = A.ravel()[0]
            # r = jnp.abs(x0)
            # theta = jnp.angle(x0)
            # vp = x +



            # P = qr.form_dense_P([v, beta])
            # Pd = dag(P)
            # unitary1 = jnp.dot(P, Pd)
            # unitary2 = jnp.dot(Pd, P)
            # Id = jnp.eye(v.size, dtype=P.dtype)
            # err1, errormsg1 = errstring(unitary1, "P Pd", Id, "I")
            # err2, errormsg2 = errstring(unitary2, "Pd P", Id, "I")
            # errormsg = ""
            # passed = True
            # if err1 > thresh:
                # passed = False
                # errormsg += "\n" + errormsg1
            # if err2 > thresh:
                # passed = False
                # errormsg += "\n" + errormsg2
            # self.assertTrue(passed, msg=errormsg)
        # self.iterloop(impl)



class TestComputeAndApplyHouseholderReflectors(GaussianMatrixTest):
    """
    Tests the code to compute and apply Householder reflections.
    """
    def __init__(self, *args, **kwargs):
        ns = (1, 6, 1)
        ms = (1, 6, 1)
        super().__init__(*args, ns=ns, ms=ms, **kwargs)

    def test_house_leftmult(self, thresh=1E-6):
        """
        Random (m,n) matrices A are generated, along with length-m vectors x.
        Householder reflections
        (v, beta) are computed from each x. The dense matrix
        P = I_m - beta v otimes dag(v) is formed. It is confirmed
        that P A and house_leftmult(A, v, beta) yield the same result.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            x = matutils.gaussian_random(shape=(m,), dtype=dtype)
            v, beta = qr.house(x)
            PA_h = qr.house_leftmult(A, v, beta)
            P = qr.form_dense_P([v, beta])
            PA = jnp.dot(P, A)
            err, errmsg = errstring(PA, "PA", PA_h, "PA_h")
            self.assertTrue(err < thresh, msg=errmsg)
        self.iterloop(impl)

    def test_house_rightmult(self, thresh=1E-6):
        """
        Random (m,n) matrices A are generated, along with length-m vectors x.
        Householder reflections
        (v, beta) are computed from each x. The dense matrix
        P = I_n - beta v otimes dag(v) is formed. It is confirmed
        that A P and house_rightmult(A, v, beta) yield the same result.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            x = matutils.gaussian_random(shape=(n,), dtype=dtype)
            v, beta = qr.house(x)
            AP_h = qr.house_rightmult(A, v, beta)
            P = qr.form_dense_P([v, beta])
            AP = jnp.dot(A, P)
            err, errmsg = errstring(AP, "AP", AP_h, "AP_h")
            self.assertTrue(err < thresh, msg=errmsg)
        self.iterloop(impl)




class GaussianQRTests(GaussianMatrixTest):
    """
    These tests check whether the QR decomposition routines function correctly,
    by generating Gaussian random input and ensuring results meet various
    conditions.

    """
    def __init__(self, *args, **kwargs):
        ns = (1, 10, 1)
        ms = (1, 10, 1)
        dtypes = [jnp.float32, jnp.complex64]
        super().__init__(*args, ns=ns, ms=ms, dtypes=dtypes, **kwargs)


    def test_forward_vs_backward_accumulation(self, thresh=1E-6):
        """
        Checks that Q computed from the factored representation gives the
        same result when using either forward or backward accumulation.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            if n > m:
                with self.assertRaises(NotImplementedError):
                    H, betas = qr.house_qr(A, mode="factored")
                return
            H, betas = qr.house_qr(A, mode="factored")
            Im = jnp.eye(m, dtype=A.dtype)
            Qforward = qr.factored_rightmult(Im, H, betas)
            Qbackward, R = qr.factored_to_QR(H, betas)
            err, errmsg = errstring(Qforward, "Qforward", Qbackward,
                                    "Qbackward")
            self.assertTrue(err < thresh, msg=errmsg)
        self.iterloop(impl)

    def test_factored_mult(self, thresh=1E-5):
        """
        A = QR -> [H, tau] is computed. R is extracted. We compare
        C * A with C * Q * R without forming Q explicitly.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup

            C = matutils.gaussian_random(shape=(n, m), dtype=dtype)
            if n > m:
                with self.assertRaises(NotImplementedError):
                    H, betas = qr.house_qr(A, mode="factored")
                return
            H, betas = qr.house_qr(A, mode="factored")
            R = jnp.triu(H)

            CA = jnp.dot(C, A)
            CQ = qr.factored_rightmult(C, H, betas)
            CQR = CQ@R
            err, errmsg = errstring(CA, "CA", CQR, "CQR")
            self.assertTrue(err < thresh, msg=errmsg)
        self.iterloop(impl)

    def test_factored_to_dense_Q(self, thresh=1E-6):
        """
        Runs the qr decomposition in 'factored' mode. Factored mode returns
        matrices H and tau that record the Householder transformations
        from which Q and R are formed.

        Specifically, R is the upper triangle of H, the Householder vectors
        mapping A to R are the lower triangle, and the normalizations of those
        vectors in a certain sense are stored in tau.

        This routine explicitly forms
        Q from these outputs using qr.factored_to_Q, checks that Q is
        unitary, and that QR = A
        to within Frobenius norm 'thresh'.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            if n > m:
                with self.assertRaises(NotImplementedError):
                    H, betas = qr.house_qr(A, mode="factored")
                return
            H, betas = qr.house_qr(A, mode="factored")
            jaxQ, jaxR = qr.factored_to_QR(H, betas)
            Id = jnp.eye(jaxQ.shape[0], dtype=A.dtype)
            errormsg = ""
            success = True

            unitary_check1 = jnp.dot(jaxQ, dag(jaxQ))
            error1, errormsg1 = errstring(unitary_check1, "Q Qdag", Id, "I")
            if error1 > thresh:
                success = False
                errormsg += "Q wasn't unitary. \n" + errormsg1 + "\n"

            unitary_check2 = jnp.dot(dag(jaxQ), jaxQ)
            error2, errormsg2 = errstring(unitary_check2, "Qdag Q", Id, "I")
            if error2 > thresh:
                success = False
                errormsg += "Q wasn't unitary. \n" + errormsg2 + "\n"

            nullopcheck = jnp.dot(jaxQ, jaxR)
            error3, errormsg3 = errstring(nullopcheck, "QR", A, "A")
            if error3 > thresh:
                errormsg += "QR != A. \n" + errormsg3 + "\n"
                success = False

            self.assertTrue(success, msg=errormsg)
        self.iterloop(impl)

    def test_WY_Q_properties(self, thresh=1E-6):
        """
        Runs the qr decomposition in 'WY' mode. WY mode returns
        matrices W and Y, storing the same Householder transformations as
        'factored' mode in a 'blocked' representation permitting their
        application using Level 3 BLAS operations.

        This routine explicitly forms
        Q from these outputs using qr.factored_to_Q. It checks that Q is
        unitary, and that QR = A
        to within Frobenius norm 'thresh'.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            if n > m:
                with self.assertRaises(NotImplementedError):
                    H, betas = qr.house_qr(A, mode="WY")
                return
            W, YH, _ = qr.house_qr(A, mode="WY")
            jaxQ = qr.WY_to_Q(W, YH)
            jaxQdag = dag(jaxQ)
            QQdag = jaxQ @ jaxQdag
            Id = jnp.eye(QQdag.shape[0], dtype=QQdag.dtype)
            err, errmsg = errstring(QQdag, "Qdag", Id, "I")
            self.assertLessEqual(err, thresh, msg=errmsg)

        self.iterloop(impl)

    def test_WY_reconstruction(self, thresh=1E-6):
        """
        Runs the qr decomposition in 'WY' mode. WY mode returns
        matrices W and Y, storing the same Householder transformations as
        'factored' mode in a 'blocked' representation permitting their
        application using Level 3 BLAS operations.

        This routine explicitly forms
        Q from these outputs using qr.factored_to_Q. It checks that Q is
        unitary, and that QR = A
        to within Frobenius norm 'thresh'.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            if n > m:
                with self.assertRaises(NotImplementedError):
                    H, betas = qr.house_qr(A, mode="WY")
                return
            W, YH, R = qr.house_qr(A, mode="WY")
            Q = qr.WY_to_Q(W, YH)
            A_recon = Q @ R
            err, errmsg = errstring(A, "A", A_recon, "QR")
            self.assertLessEqual(err, thresh, msg=errmsg)

        self.iterloop(impl)

    def test_WY_to_Q(self, thresh=1E-6):
        """
        Makes sure that retrieval of Q from WY^H, Q = I - WY^H, works
        correctly.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            W = A
            YH = matutils.gaussian_random(shape=(n, m), dtype=dtype)
            Id = jnp.eye(m, dtype=dtype)
            Q = Id - W @ YH
            Q2 = qr.WY_to_Q(W, YH)
            err, errmsg = errstring(Q, "Q", Q2, "I-WY^H")
            self.assertLessEqual(err, thresh, msg=errmsg)

    def test_B_times_Q_WY(self, thresh=1E-6):
        """
        Makes sure that B * Q = B * (I - W Y^H) for Q = I - WY^H, where
        the RHS is computed implicitly from W and YH.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            W = A
            YH = matutils.gaussian_random(shape=(n, m), dtype=dtype)
            B = matutils.gaussian_random(shape=(m, n), dtype=dtype)
            Id = jnp.eye(m, dtype=dtype)
            Q = Id - W @ YH
            BQ = B@Q
            BQ_WY = qr.B_times_Q_WY(B, W, YH)
            err, errmsg = errstring(BQ, "BQ", BQ_WY, "B(I-WY^T)")
            self.assertLessEqual(err, thresh, msg=errmsg)

class TestRandSVD(GaussianMatrixTest):
    """
    Tests of the randSVD decomposition that loop over Gaussian random matrices.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_randSVD(self, thresh=1E-2, k=5):
        """
        Tests the singular values from utv.randSVD against the numpy
        implementation.
        """
        def impl(A, paramtup):
            out_jnp = jnp.linalg.svd(A)
            out_rand = utv.randSVD(A, k=k)
            svd_jnp = out_jnp[1][:k]
            svd_rand = out_rand[1]
            error, errormsg = errstring(svd_jnp, "Numpy SVs", svd_rand,
                                        "randSVs")
            self.assertTrue(error < thresh, msg=errormsg)
        self.iterloop(impl)

    def test_randSVD_reconstruction(self, thresh=1E-3):
        """
        Checks that randSVD correctly reconstructs its input.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            U, S, Vh = utv.randSVD(A)
            A_recon = matutils.trimultdag(U, S, Vh)
            error, errormsg = errstring(A, "Input A", A_recon, "randSVD A")
            self.assertTrue(error < thresh, msg=errormsg)
        self.iterloop(impl)


class TestUTV_thin(GaussianMatrixTest):
    """
    Tests of the UTV decomposition that loop over Gaussian random matrices.

    The matrices in these tests are all "thin" (n <= m), which is assumed
    by the 'slow' algorithm.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        ns = (1, 4, 1)
        ms = (3, 6, 1)
        dtypes = [jnp.float32, jnp.complex64]
        self.ns = range(*ns)
        self.ms = range(*ms)
        self.dtypes = dtypes
        self.matrices = [matutils.gaussian_random(shape=(m, n), dtype=dtype)
                         for m, n, dtype
                         in itertools.product(self.ms, self.ns, self.dtypes)]

    ###########################################################################
    # stepUTV
    ###########################################################################
    def test_stepUTV_slow_svs(self, thresh=1E-5):
        """
        Tests the singular values from utv.stepUTV_slow against those from a
        numpy SVD.
        """
        def impl(A, paramtup):
            out_jnp = jnp.linalg.svd(A)
            out_rand = utv.stepUTV_slow(A)
            svd_sv = out_jnp[1]
            utv_sv = jnp.diag(out_rand[1])
            error, errormsg = errstring(svd_sv, "Numpy SVs", utv_sv,
                                        "rand_UTV SVs")
            self.assertTrue(error < thresh, msg=errormsg)
        self.iterloop(impl)

    def test_stepUTV_reconstruction(self, thresh=1E-5):
        """
        Checks that stepUTV correctly reconstructs its input.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            U, T, V = utv.stepUTV_slow(A)
            A_UTV = matutils.trimultmat(U, T, dag(V))
            error, errormsg = errstring(A, "Input A", A_UTV, "stepUTV A")
            self.assertTrue(error < thresh, msg=errormsg)
        self.iterloop(impl)

    ###########################################################################
    # randUTV_slow
    ###########################################################################
    def test_randUTVslow_svs(self, thresh=1E-5):
        """
        Tests the singular values from randUTV against those from
        stepUTV_slow, using blocksize = number of columns.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            out_slow = utv.stepUTV_slow(A)
            out_fast = utv.randUTV_slow(A, n, 1)
            slow_sv = out_slow[1]
            fast_sv = out_fast[1]
            error, errormsg = errstring(slow_sv, "slow UTV SVs", fast_sv,
                                        "rand_UTV SVs")
            self.assertTrue(error < thresh, msg=errormsg)
        self.iterloop(impl)

    def test_randUTVslow_reconstruction(self, thresh=1E-5):
        """
        Tests that A can be recovered from randUTV_slow, using various
        blocksizes.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            for b in range(1, n+1, 1):
                #print("m=", m, "n=", n, "b=", b, dtype)
                U, T, V = utv.randUTV_slow(A, b, 1)
                A_UTV = matutils.trimultmat(U, T, dag(V))
                error, errormsg = errstring(A, "A", A_UTV,
                                            "UTV A")
                Us, Ds, Vhs = jnp.linalg.svd(A)
                # print("U: \n", U)
                # print("U svd: \n", Us)
                # print("V: \n", dag(V))
                # print("V svd: \n", Vhs)
                # print("SVDS:", Ds)
                # print("Error: ", error)
                # print("***")
                with self.subTest(b=b):
                    self.assertTrue(error < thresh, msg=errormsg)
        self.iterloop(impl)

    ###########################################################################
    # randUTV
    ###########################################################################
    def test_randUTV_svs(self, thresh=1E-5):
        """
        Tests the singular values from randUTV against those from
        stepUTV_slow, using blocksize = number of columns.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            out_slow = utv.stepUTV_slow(A)
            out_fast = utv.randUTV(A, b=n, q=1)
            slow_sv = out_slow[1]
            fast_sv = out_fast[1]
            error, errormsg = errstring(slow_sv, "slow UTV SVs", fast_sv,
                                        "rand_UTV SVs")
            self.assertTrue(error < thresh, msg=errormsg)
        self.iterloop(impl)

    def test_randUTV_reconstruction(self, thresh=1E-5):
        """
        Tests that A can be recovered from randUTV_slow, using various
        blocksizes.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            for b in range(1, n+1, 1):
                U, T, V = utv.randUTV(A, b=n, q=1)
                A_UTV = matutils.trimultmat(U, T, dag(V))
                error, errormsg = errstring(A, "A", A_UTV,
                                            "UTV A")
                Us, Ds, Vhs = jnp.linalg.svd(A)
                with self.subTest(b=b):
                    self.assertTrue(error < thresh, msg=errormsg)
        self.iterloop(impl)


class TestUtils(GaussianMatrixTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_replace_diagonal(self, thresh=1E-6):
        def impl(A, paramtup):
            m, n, dtype = paramtup
            for i in range(1, n):
                D = matutils.gaussian_random(shape=(i,), dtype=dtype)
                with self.subTest(i=i):
                    result = matutils.replace_diagonal(A, D)
                    err = matutils.frob(jnp.triu(result, k=1),
                                        jnp.triu(A, k=1))
                    err += matutils.frob(jnp.tril(result, k=-1),
                                         jnp.tril(A, k=-1))
                    err += matutils.frob(D, jnp.diag(result)[:i])
                    err += matutils.frob(jnp.diag(result)[i:],
                                         jnp.zeros(jnp.diag(result)[i:].shape))
                    errstr = "\nA: \n " + str(A) + "\nRes: \n " + str(result)
                    errstr += "\nD: \n " + str(D)
                    errstr += "\nErr: " + str(err)
                    self.assertTrue(err < thresh, msg=errstr)
        self.iterloop(impl)

###############################################################################
# Functions to call tests
###############################################################################


def suite():
    suite = unittest.TestSuite()
    # suite.addTests(unittest.makeSuite(TestUtils, 'test'))
    # suite.addTests(unittest.makeSuite(TestUTV_thin, 'test'))
    # suite.addTests(unittest.makeSuite(TestRandSVD, 'test'))
    # suite.addTests(unittest.makeSuite(TestHouseholderVectorProperties, 'test'))
    # suite.addTests(unittest.makeSuite(TestComputeAndApplyHouseholderReflectors,
                                      # 'test'))
    # 'ExplicitQRTests' is commented out because it is wrong.
    # suite.addTests(unittest.makeSuite(ExplicitQRTests, 'test'))
    suite.addTests(unittest.makeSuite(GaussianQRTests, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main()
