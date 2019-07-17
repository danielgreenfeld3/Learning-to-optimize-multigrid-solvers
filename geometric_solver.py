import numpy as np
from warnings import warn
from scipy.sparse import csr_matrix, isspmatrix_csr, SparseEfficiencyWarning

from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers


# similar to "ruge_stuben_solver" in pyamg
def geometric_solver(A, prolongation_function, prolongation_args,
                     presmoother=('gauss_seidel', {'sweep': 'forward'}),
                     postsmoother=('gauss_seidel', {'sweep': 'forward'}),
                     max_levels=10, max_coarse=10, **kwargs):
    """Create a multilevel solver using geometric AMG.

    Parameters
    ----------
    A : csr_matrix
        Square matrix in CSR format
    presmoother : string or dict
        Method used for presmoothing at each level.  Method-specific parameters
        may be passed in using a tuple, e.g.
        presmoother=('gauss_seidel',{'sweep':'symmetric}), the default.
    postsmoother : string or dict
        Postsmoothing method with the same usage as presmoother
    max_levels: integer
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: integer
        Maximum number of variables permitted on the coarse grid.

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    Notes
    -----
    "coarse_solver" is an optional argument and is the solver used at the
    coarsest grid.  The default is a pseudo-inverse.  Most simply,
    coarse_solver can be one of ['splu', 'lu', 'cholesky, 'pinv',
    'gauss_seidel', ... ].  Additionally, coarse_solver may be a tuple
    (fn, args), where fn is a string such as ['splu', 'lu', ...] or a callable
    function, and args is a dictionary of arguments to be passed to fn.
    See [2001TrOoSc]_ for additional details.


    References
    ----------
    .. [2001TrOoSc] Trottenberg, U., Oosterlee, C. W., and Schuller, A.,
       "Multigrid" San Diego: Academic Press, 2001.  Appendix A

    See Also
    --------
    aggregation.smoothed_aggregation_solver, multilevel_solver,
    aggregation.rootnode_solver

    """
    levels = [multilevel_solver.level()]

    # convert A to csr
    if not isspmatrix_csr(A):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR",
                 SparseEfficiencyWarning)
        except BaseException:
            raise TypeError('Argument A must have type csr_matrix, \
                             or be convertible to csr_matrix')
    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels[-1].A = A

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        extend_hierarchy(levels, prolongation_function, prolongation_args)

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


# internal function
def extend_hierarchy(levels, prolongation_fn, prolongation_args):
    """Extend the multigrid hierarchy."""

    A = levels[-1].A

    # Generate the interpolation matrix that maps from the coarse-grid to the
    # fine-grid
    P = prolongation_fn(A, prolongation_args)

    # Generate the restriction matrix that maps from the fine-grid to the
    # coarse-grid
    R = P.T.tocsr()

    levels[-1].P = P  # prolongation operator
    levels[-1].R = R  # restriction operator

    levels.append(multilevel_solver.level())

    # Form next level through Galerkin product
    A = R * A * P
    A = A.astype(np.float64)  # convert from complex numbers, should have A.imag==0
    levels[-1].A = A
