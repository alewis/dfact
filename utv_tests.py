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
        """
        test_me = self.testA
        H, Tau = qr.house_qr(test_me, mode="factored")
        correct_tau = jnp.array([2./3., 8./5.])
        self.assertTrue(jnp.allclose(Tau, correct_tau),
                        msg="Tau="+str(Tau)+" was wrong.")
        correct_H = jnp.array([[3., 2.],
                               [-1., 5.],
                               [-1., 0.5]])
        self.assertTrue(jnp.allclose(H, correct_H),
                        msg="\nH=\n"+str(H)+" was wrong.")


class GaussianHouseholderVectorTests(GaussianMatrixTest):
    """
    Tests the code to compute and apply Householder reflections.
    """
    def __init__(self, *args, **kwargs):
        ns = (1, 1, 1)
        ms = (1, 5, 1)
        super().__init__(*args, ns=ns, ms=ms, **kwargs)

    def test_householder_unitarity(self, thresh=1E-7):
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
            unitary1 = jnp.dag(P, Pd)
            unitary2 = jnp.dag(Pd, P)
            Id = jnp.eye(v.size[0], dtype=P.dtype)
            err1, errormsg1 = errstring(unitary1, "P Pd", Id, "I")
            err2, errormsg2 = errstring(unitary2, "Pd P", Id, "I")
            errormsg = ""
            passed = True
            if err1:
                passed = False
                errormsg += "\n" + errormsg1
            if err2:
                passed = False
                errormsg += "\n" + errormsg2
            self.assertTrue(passed, msg=errormsg)
        self.iterloop(impl)


class GaussianHouseholderMatrixTests(GaussianMatrixTest):
    """
    Tests the code to compute and apply Householder reflections.
    """
    def __init__(self, *args, **kwargs):
        ns = (1, 5, 1)
        ms = (1, 5, 1)
        super().__init__(*args, ns=ns, ms=ms, **kwargs)

    def test_apply_house(self, thresh=1E-6):
        """
        Random (m,n) matrices A are generated, along with length-m vectors x.
        Householder reflections
        (v, beta) are computed from each x. The dense matrix
        P = I_m - beta v otimes dag(v) is formed. It is confirmed
        that P A and apply_house(A, v, beta) yield the same result.
        """
        def impl(A, paramtup):
            m, n, dtype = paramtup
            x = matutils.gaussian_random(shape=(m,), dtype=dtype)
            v, beta = qr.house(x)
            PA_h = qr.apply_house(A, v, beta)
            P = qr.form_dense_P([v, beta])
            PA = jnp.dot(P, A)
            err, errmsg = errstring(PA, "PA", PA_h, "PA_h")
            self.assertTrue(err<thresh, msg=errmsg)
        self.iterloop(impl)



    # def test_house_house_inv(self, thresh=1E-7):
        # """
        # Makes random A and x, computes h, beta = house(A) and
        # h_i, beta_i = houseInv(A), and checks that
        # applying both in sequence recovers A.
        # """

        # def impl(A, paramtup):
            # m, n, dtype = paramtup
            # x = matutils.gaussian_random(shape=(m,), dtype=dtype)
            # hvec, beta = qr.house(x)
            # h_invvec, betainv = qr.houseInv(x)
            # A_house = qr.apply_house(A, hvec, beta)
            # A_h_hI = qr.apply_house(A_house, h_invvec, betainv)
            # error, errormsg = errstring(A, "A", A_h_hI, "A_h_hI")
            # self.assertTrue(error < thresh, msg=errormsg)
        # self.iterloop(impl)







class GaussianQRTests(GaussianMatrixTest):
    """
    These tests check whether the QR decomposition routines function correctly,
    by generating Gaussian random input and ensuring results meet various
    conditions.

    """
    def __init__(self, *args, **kwargs):
        ns = (1, 5, 1)
        super().__init__(*args, ns=ns, **kwargs)

    def test_factored_to_QR(self, thresh=1E-7):
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
        to within Frobenius norm 'thresh'.  """
        def impl(A, paramtup):
            H_jax, tau_jax = qr.house_qr(A, mode="factored")
            jaxR = jnp.triu(H_jax)
            jaxQ = qr.factored_to_Q(H_jax, tau_jax)
            print(A.shape, jaxQ.shape)
            Id = jnp.eye(jaxQ.shape[0], dtype=A.dtype)
            errormsg = ""
            success = True

            unitary_check1 = jnp.dot(jaxQ, dag(jaxQ))
            error1, errormsg1 = errstring(unitary_check1, "Q Qdag", Id, "I")
            if error1 < thresh:
                success = False
                errormsg += (errormsg1 + "\n")

            unitary_check2 = jnp.dot(dag(jaxQ), jaxQ)
            error2, errormsg2 = errstring(unitary_check2, "Qdag Q", Id, "I")
            if error2 < thresh:
                success = False
                errormsg += (errormsg2 + "\n")

            nullopcheck = jnp.dot(jaxQ, jaxR)
            error3, errormsg3 = errstring(nullopcheck, "QR", A, "A")
            if error3 < thresh:
                success = False
                errormsg += (errormsg3 + "\n")

            self.assertTrue(success, msg=errormsg)
        self.iterloop(impl)

    # def test_factored_mult(self, thresh=1E-7):
        # """
        # Runs the qr decomposition in 'factored' mode, as described in the
        # docstring above.

        # This routine computes B = A v, where v is a randomly generated
        # vector, and compares the result against Btest = A_QR v, where
        # A_QR is A in its factored QR representation. The results are
        # required to agree to within a Frobenius norm of 'thresh'.
        # """
        # def impl(A, paramtup):
            # m, n, dtype = paramtup
            # H_jax, tau_jax = qr.house_qr(A, mode="factored")
            # v = gaussian_random(shape=(n,), dtype=dtype)

            # B = jnp.dot(A, v)
            # Btest = 0

            # error, errormsg = errstring(B, "A*v", Btest, "(A_QR)*v")
            # self.assertTrue(error < thresh, msg=errormsg)
        # self.iterloop(impl)

    # def test_WY_QR(self, thresh=1E-7):
        # """
        # Runs the qr decomposition in 'WY' mode. WY mode returns
        # matrices W and Y, storing the same Householder transformations as
        # 'factored' mode in a 'blocked' representation permitting their
        # application using Level 3 BLAS operations.

        # This routine explicitly forms
        # Q from these outputs using qr.factored_to_Q. It checks that Q is
        # unitary, and that QR = A
        # to within Frobenius norm 'thresh'.
        # """
        # def impl(A, paramtup):

            # W_jax, Y_jax = qr.house_qr(A, mode="WY")
            # jaxQ, jaxR = qr.WY_to_QR(W_jax, Y_jax)
            # unitary_check1 = jnp.dot(jaxQ, dag(jaxQ))
            # unitary_check2 = jnp.dot(dag(jaxQ), jaxQ)
            # I = jnp.eye(jaxQ.shape[0], dtype=A.dtype)

            # error, errormsg = errstring(unitary_check1, "Q Qdag", I, "I")
            # self.assertTrue(error < thresh, msg=errormsg)

            # error2, errormsg2 = errstring(unitary_check2, "Qdag Q", I, "I")
            # self.assertTrue(error2 < thresh, msg=errormsg2)

            # nullopcheck = jnp.dot(jaxQ, jaxR)
            # error3, errormsg3 = errstring(nullopcheck, "QR", A, "A")
            # self.assertTrue(error3 < thresh, msg=errormsg3)
        # self.iterloop(impl)
    # def test_apply_house(self, thresh=1E-7):
        # # """
        # # Tests that apply_house has the same effect as explicitly forming H
        # # and multiplying by it.
        # # """
        # def impl(A, paramtup):
            # m, n = A
            # A = gaussian_random(shape=
            # beta = gaussian_random(shape=(1), dtype=v.dtype)[0]

            # Hterm = beta * jnp.outer(v, dag(v))
            # H = jnp.eye(Hterm.shape, dtype=Hterm.dtype) - Hterm
        # params = itertools.product(self.ns, self.dtypes)
        # for A, paramtup in zip(self.matrices, params):
            # m, n, dtype = paramtup
            # with self.subTest(n=n, dtype=dtype):
                # impl(A, paramtup)

# class GaussianVectorTests(GaussianMatrixTest):
    # def setUp():
        # GaussianMatrixTest.__setUp(ns=(1, 1, 1))


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

    def test_replace_diagonal(self, thresh=1E-7):
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
    #suite.addTests(unittest.makeSuite(TestUtils, 'test'))
    #suite.addTests(unittest.makeSuite(TestUTV_thin, 'test'))
    #suite.addTests(unittest.makeSuite(TestRandSVD, 'test'))
    suite.addTests(unittest.makeSuite(GaussianHouseholderVectorTests, 'test'))
    suite.addTests(unittest.makeSuite(GaussianHouseholderMatrixTests, 'test'))
    suite.addTests(unittest.makeSuite(ExplicitQRTests, 'test'))
    # suite.addTests(unittest.makeSuite(GaussianQRTests, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main()
