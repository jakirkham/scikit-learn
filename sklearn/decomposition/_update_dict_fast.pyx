cimport libc
cimport libc.math
cimport libc.stdio
from libc.math cimport sqrt, log as ln
from libc.stdio cimport EOF, fflush, putc, puts, stdout

cimport cython
from cython cimport floating

cimport numpy as np
import numpy as np
from numpy cimport npy_intp

from sklearn.utils import check_random_state


np.import_array()


ctypedef np.uint32_t UINT32_t


cdef extern from "numpy/arrayobject.h":
    bint PyArray_IS_F_CONTIGUOUS(ndarray)


cdef inline np.ndarray ensure_fortran(np.ndarray arr, bint copy=False):
    if copy or not PyArray_IS_F_CONTIGUOUS(arr):
        return np.PyArray_NewCopy(arr, np.NPY_FORTRANORDER)
    else:
        return arr


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline void our_rand_float(UINT32_t* seed, floating* out) nogil:
    cdef int i, n
    cdef floating x

    if floating is float:
        n = 1
    else:
        n = 2

    x = 0
    for i in range(n):
        x += our_rand_r(seed)
        x /= (<UINT32_t>RAND_R_MAX + 1)

    out[0] = x


cdef inline void our_randn(UINT32_t* seed,
                           floating* out1,
                           floating* out2) nogil:
    cdef int i
    cdef floating s
    cdef floating x[2]

    while True:
        s = 0
        for i in range(2):
            our_rand_float(seed, &x[i])

            x[i] *= 2
            x[i] -= 1

            s += x[i] ** 2

        if 0 < s < 1:
            break

    s = sqrt((-2 * ln(s)) / s)

    x[0] *= s
    x[1] *= s

    out1[0] = x[0]
    out2[0] = x[1]


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef inline void randn_atom_k(UINT32_t* seed,
                              npy_intp k,
                              floating[:, :] a) nogil:
    cdef npy_intp i, r, n
    cdef floating tmp

    n = a.shape[0]
    r = n % 2

    for i in range(0, n - r, 2):
        our_randn(seed, &a[i, k], &a[i + 1, k])
    if r == 1:
        our_randn(seed, &a[n - 1, k], &tmp)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef inline void clip_negative_k(bint pos, npy_intp k, floating[:, :] a) nogil:
    if not pos:
        return

    cdef npy_intp i
    for i in range(a.shape[0]):
        if a[i, k] < 0.0:
            a[i, k] = 0.0


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef inline void norm_arr_k(npy_intp k, floating[:, :] a, floating s) nogil:
    cdef npy_intp i
    for i in range(a.shape[0]):
        a[i, k] /= s


cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        AtlasConj=114

    void dgemm "cblas_dgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, int M, int N,
                             int K, double alpha, double *A,
                             int lda, double *B, int ldb,
                             double beta, double *C, int ldc) nogil
    void sgemm "cblas_sgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, int M, int N,
                             int K, float alpha, float *A,
                             int lda, float *B, int ldb,
                             float beta, float *C, int ldc) nogil
    void dgemv "cblas_dgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                             int M, int N, double alpha, double *A, int lda,
                             double *X, int incX, double beta,
                             double *Y, int incY) nogil
    void sgemv "cblas_sgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                             int M, int N, float alpha, float *A, int lda,
                             float *X, int incX, float beta,
                             float *Y, int incY) nogil
    void dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                           double *X, int incX, double *Y, int incY,
                           double *A, int lda) nogil
    void sger "cblas_sger"(CBLAS_ORDER Order, int M, int N, float alpha,
                           float *X, int incX, float *Y, int incY,
                           float *A, int lda) nogil
    double dnrm2 "cblas_dnrm2"(int N, double *X, int incX) nogil
    float snrm2 "cblas_snrm2"(int N, float *X, int incX) nogil


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
def update_dict(np.ndarray[floating, ndim=2] dictionary not None,
                np.ndarray[floating, ndim=2] Y not None,
                np.ndarray[floating, ndim=2] code not None,
                unsigned int verbose=0, bint return_r2=False,
                random_state=None, bint positive=False):
    """Update the dense dictionary factor in place.

    Parameters
    ----------
    dictionary : array of shape (n_features, n_components)
        Value of the dictionary at the previous iteration.

    Y : array of shape (n_features, n_samples)
        Data matrix.

    code : array of shape (n_components, n_samples)
        Sparse coding of the data against which to optimize the dictionary.

    verbose:
        Degree of output the procedure will print.

    return_r2 : bool
        Whether to compute and return the residual sum of squares corresponding
        to the computed solution.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    positive : boolean, optional
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    Returns
    -------
    dictionary : array of shape (n_features, n_components)
        Updated dictionary.

    """

    # Get BLAS functions
    if floating is float:
        gemm = sgemm
        gemv = sgemv
        ger = sger
        nrm2 = snrm2
    else:
        gemm = dgemm
        gemv = dgemv
        ger = dger
        nrm2 = dnrm2

    # Get bounds
    cdef npy_intp n_features
    cdef npy_intp n_samples
    cdef npy_intp n_components

    # For holding the kth atom
    cdef floating atom_norm

    # Random number seed for use in C
    cdef UINT32_t rand_r_state_seed
    cdef UINT32_t* rand_r_state

    # Indices to iterate over
    cdef npy_intp i, j, k

    # Fortran array views/copies of the data
    cdef floating[::, :] dictionary_F
    cdef floating[::, :] code_F

    # Residuals
    cdef floating R2
    cdef floating[::, :] R

    # Verbose message
    cdef char* msg

    # Create random number seed to use
    random_state = check_random_state(random_state)
    rand_r_state_seed = random_state.randint(0, RAND_R_MAX)
    rand_r_state = &rand_r_state_seed

    # Initialize Fortran Arrays
    dictionary_F = ensure_fortran(dictionary)
    code_F = ensure_fortran(code)
    R = ensure_fortran(Y, copy=True)

    with nogil:
        # Determine verbose message
        if verbose == 0:
            msg = NULL
        elif verbose == 1:
            msg = b"+"
        else:
            msg = b"Adding new random atom"

        # Assign bounds
        n_features = Y.shape[0]
        n_components = code_F.shape[0]
        n_samples = Y.shape[1]

        # R <- -1.0 * U * V^T + 1.0 * Y
        gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
             n_features, n_samples, n_components,
             -1.0, &dictionary_F[0, 0], n_features,
             &code_F[0, 0], n_components,
             1.0, &R[0, 0], n_features)

        for k in range(n_components):
            # R <- 1.0 * U_k * V_k^T + R
            ger(CblasColMajor, n_features, n_samples,
                1.0, &dictionary_F[0, k], 1,
                &code_F[k, 0], code_F.shape[0],
                &R[0, 0], n_features)

            # U_k <- 1.0 * R * V_k^T
            gemv(CblasColMajor, CblasNoTrans,
                 n_features, n_samples,
                 1.0, &R[0, 0], n_features,
                 &code_F[k, 0], code_F.shape[0],
                 0.0, &dictionary_F[0, k], 1)

            # Clip negative values
            clip_negative_k(positive, k, dictionary_F)

            # Scale k'th atom
            # (U_k * U_k) ** 0.5
            atom_norm = nrm2(dictionary_F.shape[0], &dictionary_F[0, k], 1)

            # Generate random atom to replace inconsequential one
            if atom_norm < 1e-10:
                # Handle verbose mode
                if msg is not NULL:
                    if puts(msg) == EOF or fflush(stdout) == EOF:
                        with gil:
                            raise IOError("Failed to print out state.")

                # Seed random atom
                randn_atom_k(rand_r_state, k, dictionary_F)

                # Clip negative values
                clip_negative_k(positive, k, dictionary_F)

                # Setting corresponding coefs to 0
                for j in range(code_F.shape[1]):
                    code_F[k, j] = 0.0

                # Compute new norm
                # (U_k * U_k) ** 0.5
                atom_norm = nrm2(dictionary_F.shape[0], &dictionary_F[0, k], 1)

                # Normalize atom
                norm_arr_k(k, dictionary_F, atom_norm)
            else:
                # Normalize atom
                norm_arr_k(k, dictionary_F, atom_norm)

                # R <- -1.0 * U_k * V_k^T + R
                ger(CblasColMajor, n_features, n_samples,
                    -1.0, &dictionary_F[0, k], 1,
                    &code_F[k, 0], code_F.shape[0],
                    &R[0, 0], n_features)

        # Compute sum of squared residuals
        if return_r2:
            R2 = 0.0
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    R2 += R[i, j] ** 2

    if return_r2:
        return dictionary_F.base, R2
    else:
        return dictionary_F.base
