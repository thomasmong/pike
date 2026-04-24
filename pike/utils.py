import torch
from typing import List, Dict, Tuple

def compile_matrices(
        polynomial_definitions: List[Dict[Tuple[int, ...], float]],
        input_dim: int
    ) -> Dict[str, torch.Tensor]:
    """
    Compiles a list of symbolic polynomials into dense matrices optimized for batch evaluation.
    
    Args:
        polynomial_definitions: List of dicts. Each dict is {exponent_tuple: coefficient}.
        input_dim: The number of variables in the input space.
        
    Returns:
        A dictionary containing:
        - 'W': Weights matrix (Num_Polys x Num_Monomials)
        - 'E': Exponents matrix (Num_Monomials x Input_Dim)
    """
    # 1. Identify unique monomials
    unique_exponents = set()
    for p in polynomial_definitions:
        unique_exponents.update(p.keys())
    sorted_exponents = sorted(list(unique_exponents))
    exp_to_idx = {exp: idx for idx, exp in enumerate(sorted_exponents)}

    num_monomials = len(sorted_exponents)
    num_polys = len(polynomial_definitions)

    # 2. Build matrices
    E = torch.zeros((num_monomials, input_dim))
    W = torch.zeros((num_polys, num_monomials))

    for i, exp_tuple in enumerate(sorted_exponents):
        if len(exp_tuple) != input_dim:
            raise ValueError(f"Exponent tuple {exp_tuple} does not match input dimension {input_dim}.")
        for d, val in enumerate(exp_tuple):
            E[i, d] = val
    
    for i, p in enumerate(polynomial_definitions):
        for exp, coeff in p.items():
            idx = exp_to_idx[exp]
            W[i, idx] = coeff

    return {
        'W': W,
        'E': E
    }

def compute_monomials(
        X: torch.Tensor,
        E: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluates all unique monomials for a batch of points using broadcasting.
    M = Product(X_i ^ E_i)
    
    Args:
        X: Input batch (Input_Dim x Batch_Size)
        E: Exponents matrix (Num_Monomials x Input_Dim)
    """
    # X shape: (D, B)
    # E shape: (M, D)
    
    # We use broadcasting to compute all powers:
    # (M, D, 1) ^ (1, D, B) -> (M, D, B)
    powers = torch.pow(X.unsqueeze(0), E.unsqueeze(2))
    
    # Multiply across the Input_Dim (D) to get monomials
    # (M, D, B) -> (M, B)
    M = torch.prod(powers, dim=1)

    # Memory management
    del powers
    torch.cuda.empty_cache()
    
    return M

def evaluate_polynomials(X: torch.Tensor, matrices: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Full pipeline wrapper.
    
    Args:
        X: Input points (Input_Dim x Batch_Size)
        matrices: Dictionary output from `compile_matrices`
    """
    # 1. Get Monomials (K x P)
    M = compute_monomials(X, matrices['E'])
    
    # 2. Linear Combination (N x K) @ (K x P) -> (N x P)
    Y = torch.matmul(matrices['W'], M)
    
    return Y

# Derivative utilities
def compute_gradients(polynomial_definitions: List[Dict[Tuple[int, ...], float]], input_dim: int) -> List[List[Dict[Tuple[int, ...], float]]]:
    """
    Computes the symbolic gradients of a list of polynomials.
    
    Args:
        polynomial_definitions: List of dicts. Each dict is {exponent_tuple: coefficient}.
        input_dim: The number of variables in the input space.
        
    Returns:
        A list of lists of dicts. Outer list is per polynomial, inner list is per variable.
        Each dict is {exponent_tuple: coefficient} representing the derivative w.r.t that variable.
    """
    gradients = []
    
    for p in polynomial_definitions:
        poly_grads = []
        for d in range(input_dim):
            grad_d = {}
            for exp_tuple, coeff in p.items():
                if exp_tuple[d] > 0:
                    new_exp = list(exp_tuple)
                    new_exp[d] -= 1
                    grad_coeff = coeff * exp_tuple[d]
                    grad_d[tuple(new_exp)] = grad_coeff
            poly_grads.append(grad_d)
        gradients.append(poly_grads)
    
    return gradients