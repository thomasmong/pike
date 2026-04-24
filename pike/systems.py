from . import utils
import torch
from scipy.integrate import solve_ivp
from itertools import combinations_with_replacement

class PolyParamAffineSystem:
    """Polynomial parameter-affine dynamical system.

    Represents:
        dx/dt = f0(x) + sum_{i=1}^{p} mu_i * fi(x)

    Vector fields are given in sparse monomial form:
        f_mono[i][j] = [(exp_tuple, coeff), ...]
    where f_mono[0] is the drift and f_mono[1:] are the parameter-dependent
    fields.  Each entry f_mono[i][j] is the j-th state coordinate of fi,
    as a list of (exponent_tuple, coefficient) pairs.  An empty list []
    represents a zero component.

    Parameters
    ----------
    n_vars : int
        Number of state variables.
    degree : int
        Maximum polynomial degree (used to build the monomial basis).
    f_mono : list
        Nested list [f0, f1, ..., fp] where each fi is a list of n_vars
        polynomial components.
    """

    def __init__(self, n_vars: int, degree: int, f_mono: list, device: str = "cpu"):
        self.n_vars = n_vars
        self.degree = degree
        self.f_mono = f_mono
        self.n_params = len(f_mono) - 1
        self.exp_to_idx, self.idx_to_exp = self._create_monomial_maps()
        self.n_monomials = len(self.exp_to_idx)
        self.device = device
        
        self._matrices = self._compile_torch()

    # ------------------------------------------------------------------
    # Monomial basis
    # ------------------------------------------------------------------

    def _create_monomial_maps(self):
        monomials = []
        for d in range(self.degree + 1):
            for combo in combinations_with_replacement(range(self.n_vars), d):
                exponents = [0] * self.n_vars
                for var_idx in combo:
                    exponents[var_idx] += 1
                monomials.append(tuple(exponents))
        return (
            {exp: i for i, exp in enumerate(monomials)},
            {i: exp for i, exp in enumerate(monomials)},
        )
    
    def _compile_torch(self):
        """
        Flatten the nested list structure of f_mono into a single list of 
        polynomial definitions compatible with utils.
        """
        poly_dict = []
        for i in range(self.n_params + 1):
            for j in range(self.n_vars):
                poly_dict.append(self.f_mono[i][j])

        # Compile
        matrices = utils.compile_matrices(poly_dict, self.n_vars)
        # Move to device
        for k, v in matrices.items():
            matrices[k] = v.to(self.device).double()

        return matrices

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    
    def __call__(self, X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Evaluate f(x, mu) = f0(x) + sum_i mu_i * fi(x).
        """
            
        # 1. Evaluate all polynomial components at once
        # Output Y is shape (Total_Polys, Batch)
        # Total_Polys = (1 + n_params) * n_vars
        with torch.no_grad():
            Y = utils.evaluate_polynomials(X, self._matrices)
        
        # 3. Reshape Y to (num_fields, n_vars, batch)
        # num_fields = 1 + n_params (f0, f1, f2...)
        num_fields = 1 + self.n_params
        batch_size = Y.shape[1]
        
        Y_reshaped = Y.view(num_fields, self.n_vars, batch_size)
        
        # 4. Compute Linear Combination
        # f_drift = Y_reshaped[0]
        # f_control = Y_reshaped[1:]
        
        # Result = f0 + sum(mu_i * fi)
        result = Y_reshaped[0].clone()
        if self.n_params > 0:
            mu_broadcast = mu.view(-1, 1, 1)
            result += torch.sum(Y_reshaped[1:] * mu_broadcast, dim=0)
        del Y, Y_reshaped
        torch.cuda.empty_cache()

        return result

    # ------------------------------------------------------------------
    # Trajectory simulation
    # ------------------------------------------------------------------

    def simulate(self, x0, mu, t_eval, method="RK45", **kwargs):
        t_np = t_eval.cpu().numpy()
        def rhs(t, x):
            x_t = torch.tensor(x, device=self.device, dtype=torch.float64).unsqueeze(-1)
            return self.__call__(x_t, mu).squeeze(-1).cpu().numpy()
        sol = solve_ivp(rhs, [t_np[0], t_np[-1]], x0.cpu().numpy(),
                            t_eval=t_np, method=method, **kwargs)
        return torch.tensor(sol.y, device=self.device, dtype=torch.float64)  # (n_vars, n_samples)

    def __repr__(self) -> str:
        return (
            f"PolyParamAffineSystem("
            f"n_vars={self.n_vars}, degree={self.degree}, n_params={self.n_params})"
        )

class ClosedPoly(PolyParamAffineSystem):
    """Polynomial system with guaranteed closure under Lie derivatives.

    Constructs a PolyParamAffineSystem where the monomial basis up to the
    specified degree is closed under the Lie derivatives induced by the
    vector fields.  This is achieved by defining f_mono in terms of the
    monomial basis itself, ensuring that all Lie derivatives remain within
    the span of the basis.

    Parameters
    ----------
    n_vars : int
        Number of state variables.
    degree : int
        Maximum polynomial degree (used to build the monomial basis).
    """

    def __init__(self, n_vars: int, degree: int, device: str = "cpu"):
        self.n_vars = n_vars
        self.degree = degree
        self.exp_to_idx, self.idx_to_exp = self._create_monomial_maps()
        self.n_monomials = len(self.exp_to_idx)
        # Define f_mono to ensure closure: each component is a linear combination of monomials
        f_mono = [[[] for _ in range(n_vars)] for _ in range(n_vars + 1)]

        # f_0
        temp = []
        for j in range(n_vars):
            f_mono[0][j] = list(temp)  # snapshot before adding x_j
            exps = [0] * n_vars
            exps[j] = 1
            temp.append((tuple(exps), 1))
            # Increment exponents > 0 in all accumulated monomials
            new_temp = []
            for mono_exps, coeff in temp:
                new_exps = list(mono_exps)
                for k in range(n_vars):
                    if new_exps[k] > 0:
                        new_exps[k] += 1
                new_temp.append((tuple(new_exps), coeff))
            temp = new_temp

        # f_i (parameter-dependent terms)
        for i in range(1, n_vars + 1):
            exps = [0] * n_vars
            exps[i-1] = 1
            f_mono[i][i-1] = [(tuple(exps), 1)]
        
        # To new representation
        f_new = [[{} for _ in range(n_vars)] for _ in range(n_vars + 1)]
        for i in range(n_vars + 1):
            for j in range(n_vars):
                for exp, coeff in f_mono[i][j]:
                    exp_tuple = tuple(exp)
                    f_new[i][j][exp_tuple] = coeff

        super().__init__(n_vars, degree, f_new, device)
            
class VanDerPolSystem(PolyParamAffineSystem):
    """Van der Pol oscillator as a polynomial system.

    dx1/dt = x2
    dx2/dt = mu * (1 - x1^2) * x2 - x1

    Parameters
    ----------
    degree : int
        Maximum polynomial degree (must be >= 3 for closure).
    """

    def __init__(self, degree: int, device: str = "cpu"):
        n_vars = 2
        f_mono = [
            [{(0,1): 1}, {(1,0): -1}],
            [{}, {(0,1): 1, (2,1): -1}],
        ]
        super().__init__(n_vars, degree, f_mono, device)