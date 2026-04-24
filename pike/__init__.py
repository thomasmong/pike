"""
PIKE — Parameter-Independent Koopman Expansion
===============================================

Public API
----------
Systems:
    PolyParamAffineSystem   Base class for polynomial parameter-affine systems.
    ClosedPoly              Built-in benchmark system with guaranteed closure.
    VanDerPolSystem         Van der Pol oscillator.

Algorithm:
    PIKE                    Iterative closure procedure; generates a
                            Koopman-invariant dictionary and the associated
                            matrices K0, K1, ..., Kp.
    OutOfDictionaryError    Raised when a Lie derivative exceeds the
                            prescribed polynomial degree.

Estimation:
    KoopmanEstimation       gEDMD, iEDMD, pEDMD, empirical-pEDMD,
                            sparse-iEDMD estimators.

Typical usage
-------------
    from pike import ClosedPoly, PIKE, KoopmanEstimation

    system = ClosedPoly(n_vars=3, degree=5, device="cuda")

    pike = PIKE(system)
    psi_defs, K = pike.generate()

    ke = KoopmanEstimation(psi_defs, n_vars=3, device="cuda")
    K_mu, mu_est, _ = ke.pEDMD(K, X=X, X_dot=X_dot)
"""

from .systems import PolyParamAffineSystem, ClosedPoly, VanDerPolSystem
from .algorithm import PIKE, OutOfDictionaryError
from .estimation import KoopmanEstimation

__all__ = [
    # Systems
    "PolyParamAffineSystem",
    "ClosedPoly",
    "VanDerPolSystem",
    # Algorithm
    "PIKE",
    "OutOfDictionaryError",
    # Estimation
    "KoopmanEstimation",
]