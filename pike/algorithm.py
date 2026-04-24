from . import systems
import torch
import numpy as np

    
class OutOfDictionaryError(Exception):
    """Raised when a Lie derivative produces a monomial outside the dictionary."""
    pass

class PIKE:
    """Koopman-invariant observable dictionary for a PolyParamAffineSystem.

    Builds a dictionary D of polynomial observables closed under the Lie
    derivatives induced by every vector field in the system (drift and all
    parameter-dependent terms).

    Algorithm
    ---------
    1. Initialize D with the full-state observables [x1, ..., xn].
    2. For each h in D (including newly added entries) and each fi in f_mono:
       a. Compute the Lie derivative  L_fi(h) = grad(h) · fi(x).
       b. If L_fi(h) lies outside span(D) (residual > tol), append its
          out-of-span component to D.

    Observables are stored internally as index-based sparse polynomials:
        [(monomial_index, coefficient), ...]
    where indices refer to the monomial basis of the parent system.

    Parameters
    ----------
    system : PolyParamAffineSystem
    """

    def __init__(self, system: systems.PolyParamAffineSystem):
        self.system = system
        self.n_vars = system.n_vars
        self.exp_to_idx = system.exp_to_idx
        self.idx_to_exp = system.idx_to_exp
        self.n_monomials = system.n_monomials
        self.D: list | None = None
        self.K: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Monomial algebra (index-based)
    # ------------------------------------------------------------------

    def _mono_derivative(self, g_idx: int, var_idx: int):
        """Differentiate monomial g_idx w.r.t. variable var_idx.

        Returns (result_idx, coefficient) or (None, 0) if derivative is zero.
        """
        exp = self.idx_to_exp[g_idx]
        if exp[var_idx] == 0:
            return None, 0
        new_exp = list(exp)
        coeff = new_exp[var_idx]
        new_exp[var_idx] -= 1
        return self.exp_to_idx[tuple(new_exp)], coeff

    def _mono_gradient(self, g_idx: int) -> list:
        """Return [(result_idx, coeff)] for each partial derivative."""
        return [self._mono_derivative(g_idx, j) for j in range(self.n_vars)]

    def _add_exponents(self, idx1: int, idx2: int) -> int:
        e1 = self.idx_to_exp[idx1]
        e2 = self.idx_to_exp[idx2]
        key = tuple(a + b for a, b in zip(e1, e2))
        if key not in self.exp_to_idx:
            raise OutOfDictionaryError(
                f"Monomial {key} (degree {sum(key)}) exceeds dictionary degree {self.system.degree}"
            )
        return self.exp_to_idx[key]

    def _mono_koopman(self, g_idx: int, f: list) -> list:
        """Lie derivative of monomial g_idx along vector field f.

        f[j] = [(exp_tuple, coeff), ...] is the j-th component of f.
        Returns a sparse polynomial [(idx, coeff), ...].
        """
        result = []
        for var_idx, (d_idx, d_coeff) in enumerate(self._mono_gradient(g_idx)):
            if d_coeff == 0:
                continue
            for f_exp, f_coeff in f[var_idx].items():
                f_idx = self.exp_to_idx[f_exp]
                new_idx = self._add_exponents(d_idx, f_idx)
                result.append((new_idx, d_coeff * f_coeff))
        return result

    def _koopman(self, obs: list, f: list) -> list:
        """Lie derivative of observable obs along vector field f."""
        result = []
        for idx, coeff in obs:
            for new_idx, new_coeff in self._mono_koopman(idx, f):
                result.append((new_idx, coeff * new_coeff))
        return result

    # ------------------------------------------------------------------
    # Sparse polynomial ↔ vector conversions
    # ------------------------------------------------------------------

    def _to_vector(self, poly: list) -> np.ndarray:
        v = np.zeros(self.n_monomials)
        for idx, coeff in poly:
            v[idx] += coeff
        return v

    def _from_vector(self, v: np.ndarray, tol: float = 1e-6) -> list:
        return [(i, float(c)) for i, c in enumerate(v) if abs(c) > tol]

    def _to_matrix(self, polys: list) -> np.ndarray:
        """Stack sparse polynomials into a matrix (one row per observable)."""
        return np.array([self._to_vector(p) for p in polys])

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def _proj(self, poly: list, D: list):
        """Project poly onto span(D) in the monomial-vector space.

        Solves B.T @ c ≈ v (least squares) where B rows are observables in D.

        Returns
        -------
        coeffs : ndarray, shape (len(D),)
        residual_norm : float
        """
        B = self._to_matrix(D)           # (len(D), n_monomials)
        v = self._to_vector(poly)        # (n_monomials,)
        coeffs, *_ = np.linalg.lstsq(B.T, v, rcond=None)
        residual_norm = float(np.linalg.norm(v - coeffs @ B))
        return coeffs, residual_norm

    # ------------------------------------------------------------------
    # Dictionary generation
    # ------------------------------------------------------------------

    def generate(self, mu = None, tol: float = 1e-6) -> tuple:
        """Generate the Koopman-invariant dictionary and operator matrices.

        Parameters
        ----------
        tol : float
            Residual norm threshold for adding a new observable.

        Returns
        -------
        D : list of sparse polynomials [(monomial_idx, coeff), ...]
        Ks : ndarray, shape (len(f), len(D), len(D))
            Koopman operator matrix for each vector field in f_mono (drift
            first, then parameter-dependent terms).  _i[:, j] holds the
            coordinates of L_fi(D[j]) in the basis D.
        """
        n_vars = self.n_vars
        idx_of = self.exp_to_idx

        # Full-state observables: x1, x2, ..., xn
        D = [
            [(idx_of[tuple(1 if i == j else 0 for j in range(n_vars))], 1.0)]
            for i in range(n_vars)
        ]
        m = n_vars

        K_data = [{} for _ in range(self.system.n_params + 1)]

        closed = True

        if mu is None:
            for j, h in enumerate(D):              # D grows in place during iteration
                for (ind_f,f) in enumerate(self.system.f_mono):
                    try:
                        h_new = self._koopman(h, f)
                    except OutOfDictionaryError:
                        # Lie derivative of h along f exceeds the dictionary degree;
                        # skip this pair (no entry in K, no new observable added).
                        closed = False
                        continue
                    if h_new:
                        coeffs, res = self._proj(h_new, D)
                        # Coeffs from D
                        for k, c in enumerate(coeffs):
                            if c != 0:
                                K_data[ind_f][(j, k)] = c
                        if res > tol:
                            B = self._to_matrix(D)
                            r_vec = self._to_vector(h_new) - coeffs @ B
                            r = self._from_vector(r_vec, tol)
                            if r:
                                K_data[ind_f][(j, m)] = 1.0
                                D.append(r)
                                m += 1
        else:
            f_total = [
                list(self.system.f_mono[0][j]) + [
                    (exp, mu[i] * coeff)
                    for i in range(self.system.n_params)
                    for exp, coeff in self.system.f_mono[i + 1][j].items()
                ]
                for j in range(self.system.n_vars)
            ]
            for j, h in enumerate(D):
                try:
                    h_new = self._koopman(h, f_total)
                except OutOfDictionaryError:
                    # Lie derivative of h exceeds the dictionary degree;
                    # skip this observable (no entry in K, no new observable added).
                    closed = False
                    continue
                if h_new:
                    coeffs, res = self._proj(h_new, D)
                    for k, c in enumerate(coeffs):
                        if c != 0:
                            K_data[0][(j, k)] = c
                    if res > tol:
                        B = self._to_matrix(D)
                        r_vec = self._to_vector(h_new) - coeffs @ B
                        r = self._from_vector(r_vec, tol)
                        if r:
                            K_data[0][(j, m)] = 1.0
                            D.append(r)
                            m += 1

        self.closed = closed
        if not closed:
            print(
                f"[PIKE] Warning: closure not reached within degree "
                f"{self.system.degree}. Dictionary truncated to {m} observable(s)."
            )

        m = len(D)
        K = np.zeros((self.system.n_params + 1, m, m))
        for i, kd in enumerate(K_data):
            for (j, k), v in kd.items():
                K[i, j, k] = v
        self.D = D
        self.K = K
        return self._create_poly_dict(), torch.as_tensor(self.K, dtype=torch.float64, device=self.system.device)

    def _create_poly_dict(self):
        """Convert internal sparse format to poly_utils format."""
        if self.D is None: return

        poly_dict = []
        for obs in self.D:
            d = {}
            for idx, coeff in obs:
                exp_tuple = self.idx_to_exp[idx]
                d[exp_tuple] = coeff
            poly_dict.append(d)
        return poly_dict

    def __len__(self) -> int:
        return len(self.D) if self.D is not None else 0

    def __repr__(self) -> str:
        n = len(self) if self.D is not None else "?"
        return f"PIKE(n_vars={self.n_vars}, |D|={n})"
