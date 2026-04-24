from . import utils
import torch
import numpy as np
from sklearn.cluster import KMeans

class KoopmanEstimation:

    def __init__(self, psi_defs, input_dim, device="cpu"):
        """
        Args:
            psi_defs: List of dictionary function definitions (symbolic dicts).
            input_dim: Number of state variables.
            device: Device for torch computations.
        """
        self.psi = psi_defs
        self.n_vars = input_dim
        self.n_obs = len(psi_defs)
        self.device = device

        # 1. Compute symbolic gradients: List[List[Dict]]
        # Each outer list is an observable, inner list is [d/dx1, d/dx2, ...]
        self.grad_psi = utils.compute_gradients(psi_defs, self.n_vars)

        # 2. Compile matrices for Psi evaluation
        self._matrices = utils.compile_matrices(self.psi, self.n_vars)
        for k, v in self._matrices.items():
            self._matrices[k] = v.to(self.device).double()
        
        # 3. Flatten gradient definitions for matrix compilation
        # Order: [d_psi1/dx1, ..., d_psi1/dxn, d_psi2/dx1, ...]
        grad_psi_flat = []
        for grad_list in self.grad_psi:
            for grad_component in grad_list:
                grad_psi_flat.append(grad_component)
        
        self._grad_matrices = utils.compile_matrices(grad_psi_flat, self.n_vars)
        for k, v in self._grad_matrices.items():
            self._grad_matrices[k] = v.to(self.device).double()

    def _eval_psi(self, x_torch: torch.Tensor) -> torch.Tensor:
        """Evaluate dictionary observables: Psi(X). Output: (n_obs, T) torch tensor."""
        with torch.no_grad():
            psi_x = utils.evaluate_polynomials(x_torch, self._matrices)
        return psi_x

    def _eval_dot_psi(self, x_torch: torch.Tensor, x_dot_torch: torch.Tensor) -> torch.Tensor:
        """
        Compute d/dt Psi(X) = grad(Psi) * X_dot via vectorized Jacobian.
        Output: (n_obs, T) torch tensor.
        """
        T = x_torch.shape[1]

        with torch.no_grad():
            # 1. Evaluate all partial derivatives at once
            # Result is (n_obs * n_vars, T)
            grad_flat = utils.evaluate_polynomials(x_torch, self._grad_matrices)
            
            # 2. Reshape to (n_obs, n_vars, T)
            # This matches the flattening order in __init__
            grad_reshaped = grad_flat.view(self.n_obs, self.n_vars, T)
            
            # 3. Compute dot product: sum_i (d_psik / d_xi) * (d_xi / dt)
            # grad_reshaped: (n_obs, n_vars, T)
            # x_dot_torch: (n_vars, T) -> unsqueeze to (1, n_vars, T) for broadcasting
            dot_psi = torch.sum(grad_reshaped * x_dot_torch.unsqueeze(0), dim=1)
            
        del grad_flat, grad_reshaped
        torch.cuda.empty_cache()
        return dot_psi

    # ------------------------------------------------------------------
    # Lifting helper
    # ------------------------------------------------------------------

    def _resolve_lifted(self, X, X_dot, psi=None, dot_psi=None):
        """Return (psi, dot_psi), computing from (X, X_dot) only when needed."""
        if psi is None:
            psi = self._eval_psi(X)
        if dot_psi is None:
            dot_psi = self._eval_dot_psi(X, X_dot)
        return psi, dot_psi

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def gEDMD(self,
              X: torch.Tensor = None,
              X_dot: torch.Tensor = None,
              psi: torch.Tensor = None,
              dot_psi: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate the Koopman generator K.
        If *psi* and/or *dot_psi* are supplied they are used directly,
        otherwise they are computed from (X, X_dot).
        """
        psi_x, dot_psi_x = self._resolve_lifted(X, X_dot, psi, dot_psi)

        A = psi_x @ psi_x.T
        B = dot_psi_x @ psi_x.T
        cond = torch.linalg.cond(A.cpu())
        K = torch.linalg.solve(A, B.T).T
        del A, B
        torch.cuda.empty_cache()

        return K, cond

    def pEDMD(self,
              K: torch.Tensor,
              X: torch.Tensor = None,
              X_dot: torch.Tensor = None,
              psi: torch.Tensor = None,
              dot_psi: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parametric EDMD: optimize over the parameters mu.
        If *psi* and/or *dot_psi* are supplied they are used directly,
        otherwise they are computed from (X, X_dot).
        """
        psi_x, dot_psi_x = self._resolve_lifted(X, X_dot, psi, dot_psi)

        Mxx = psi_x @ psi_x.T
        Myx = (dot_psi_x - K[0] @ psi_x) @ psi_x.T
        d = K.shape[0] - 1

        G = torch.zeros((d, d), device=self.device, dtype=torch.float64)
        b = torch.zeros(d, device=self.device, dtype=torch.float64)

        K_params = K[1:]                                              # (d, n_obs, n_obs)
        KM = K_params @ Mxx                                           # (d, n_obs, n_obs) — 1 batched op
        G = torch.einsum('iab,jab->ij', K_params, KM)                # (d, d)           — 1 op
        b = (K_params * Myx.unsqueeze(0)).sum(dim=(-2, -1))           # (d,)             — 1 op

        cond = torch.linalg.cond(G.cpu())
        mu = torch.linalg.solve(G, b)
        K = K[0] + sum(mu[i] * K[i+1] for i in range(d))

        del Mxx, Myx, G, b, KM
        torch.cuda.empty_cache()

        return K, mu, cond
    
    # ------------------------------------------------------------------
    # Training (run once on multi-system data)
    # ------------------------------------------------------------------

    def fit_e_pEDMD(self,
                    X_train: torch.Tensor,
                    X_dot_train: torch.Tensor,
                    mus: torch.Tensor,
                    ) -> torch.Tensor:
        """
        Recover the individual matrices {K_i} from labeled multi-system data.

        Parameters
        ----------
        X_train     : (Q, n_vars, T')    State snapshots for Q training systems.
        X_dot_train : (Q, n_vars, T')    Time derivatives for Q training systems.
        mus         : (Q, P)             Parameter vectors for Q training systems.

        Returns
        -------
        K_est : (P+1, n_obs, n_obs)  Estimated [K_0, K_1, ..., K_P].
        """
        Q = X_train.shape[0]
        P = mus.shape[1]
        T_train = X_train.shape[2]

        psi = torch.zeros((Q, self.n_obs, T_train), device=self.device, dtype=torch.float64)
        psi_dot = torch.zeros_like(psi)
        for q in range(Q):
            psi[q] = self._eval_psi(X_train[q])
            psi_dot[q] = self._eval_dot_psi(X_train[q], X_dot_train[q])

        ones = torch.ones((Q, 1), device=self.device, dtype=torch.float64)
        augmented_params = torch.cat((ones, mus), dim=1)

        omega = augmented_params[:, :, None, None] * psi[:, None, :, :]
        omega = omega.view(Q, -1, T_train)

        omega_all = omega.permute(1, 0, 2).reshape((P + 1) * self.n_obs, T_train * Q)
        psi_dot_all = psi_dot.permute(1, 0, 2).reshape(self.n_obs, T_train * Q)

        K_full_T = torch.linalg.lstsq(omega_all.T, psi_dot_all.T).solution
        K_full = K_full_T.T

        del psi, psi_dot, omega, omega_all, psi_dot_all
        torch.cuda.empty_cache()

        return K_full.view(self.n_obs, P + 1, self.n_obs).permute(1, 0, 2)


    def fit_sparse_iEDMD(self,
                        X_train: torch.Tensor,
                        X_dot_train: torch.Tensor,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Identify constant vs variable entries in K from multi-system data.

        Parameters
        ----------
        X_train     : (Q, n_vars, T')    State snapshots for Q training systems.
        X_dot_train : (Q, n_vars, T')    Time derivatives for Q training systems.

        Returns
        -------
        K_0    : (n_obs, n_obs)  Mean values for constant entries (0 elsewhere).
        K_mask : (n_obs, n_obs)  Binary mask (1 = free, 0 = fixed).
        """
        Q = X_train.shape[0]

        Ks = torch.empty(Q, self.n_obs, self.n_obs, device=self.device, dtype=torch.float64)
        for q in range(Q):
            Ks[q], _ = self.gEDMD(X_train[q], X_dot_train[q])

        K_mean = Ks.mean(dim=0)
        K_std = Ks.std(dim=0)
        K_std_log = torch.log(K_std + 1e-10)

        log_std_flat = K_std_log.cpu().flatten().numpy().reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(log_std_flat)
        constant_cluster = int(np.argmin(kmeans.cluster_centers_.flatten()))
        is_constant = (kmeans.labels_ == constant_cluster).reshape(self.n_obs, self.n_obs)

        K_mask = 1 - torch.tensor(is_constant, device=self.device, dtype=torch.float64)
        K_c = K_mean * (1 - K_mask)

        del Ks, K_mean, K_std, K_std_log, log_std_flat
        torch.cuda.empty_cache()

        return K_c, K_mask

    def precompute_sparse_structure(self, K_mask: torch.Tensor):
        """Cache free/fixed indices per row. Call once after fit_sparse_iEDMD."""
        self._sparse_free  = []
        self._sparse_fixed = []
        K_mask_cpu = K_mask.cpu()
        for i in range(self.n_obs):
            self._sparse_free.append(torch.nonzero(K_mask_cpu[i] == 1, as_tuple=True)[0])
            self._sparse_fixed.append(torch.nonzero(K_mask_cpu[i] == 0, as_tuple=True)[0])

    # ------------------------------------------------------------------
    # Estimation (run on any new target system)
    # ------------------------------------------------------------------

    def e_pEDMD(self,
                K_est: torch.Tensor,
                X: torch.Tensor = None,
                X_dot: torch.Tensor = None,
                psi: torch.Tensor = None,
                dot_psi: torch.Tensor = None,
                ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Estimate mu and K for a new system given pre-computed {K_i}.

        Parameters
        ----------
        X       : (n_vars, T)             Target state snapshots.
        X_dot   : (n_vars, T)             Target time derivatives.
        K_est   : (P+1, n_obs, n_obs)     From fit_e_pEDMD.
        psi     : (n_obs, T), optional     Pre-computed lifted states.
        dot_psi : (n_obs, T), optional     Pre-computed lifted derivatives.

        Returns
        -------
        K  : (n_obs, n_obs)  Reconstructed generator.
        mu : (P,)            Estimated parameters.
        """
        return self.pEDMD(K_est, X=X, X_dot=X_dot, psi=psi, dot_psi=dot_psi)


    def sparse_iEDMD(self, K_c, K_mask, X=None, X_dot=None, psi=None, dot_psi=None):
        psi_x, dot_psi_x = self._resolve_lifted(X, X_dot, psi, dot_psi)

        # Compute on GPU, then transfer ONCE (8×8 matrices — negligible cost)
        G = (psi_x @ psi_x.T).cpu()
        V = (dot_psi_x @ psi_x.T).cpu()
        K_c_cpu = K_c.cpu()
        K = K_c_cpu.clone()

        # Use cached indices (precomputed once, never recomputed)
        rows = self._sparse_free if hasattr(self, '_sparse_free') else None
        if rows is None:
            self.precompute_sparse_structure(K_mask)

        for i in range(self.n_obs):
            free_idx  = self._sparse_free[i]
            fixed_idx = self._sparse_fixed[i]

            if len(free_idx) == 0:
                continue

            LHS     = G[free_idx][:, free_idx]
            rhs_vec = V[i, free_idx]

            if len(fixed_idx) > 0:
                rhs_vec = rhs_vec - K_c_cpu[i, fixed_idx] @ G[fixed_idx][:, free_idx]

            # torch.linalg.solve is fast on CPU for 1×1 to 3×3 systems
            K[i, free_idx] = torch.linalg.lstsq(LHS, rhs_vec).solution

        del G, V
        torch.cuda.empty_cache()

        return K.to(self.device)
    
    def __repr__(self) -> str:
        return f"Estimation(n_obs={self.n_obs}, device={self.device})"

