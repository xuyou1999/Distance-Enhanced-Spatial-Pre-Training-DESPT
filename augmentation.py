import numpy as np
import torch

def edge_masking(adj_matrix, rem, device='cpu'):
    """
    Transforms the given adjacency matrix based on the comparison with a random matrix and a threshold.
    The diagonal of the adjacency matrix will remain unchanged.

    Parameters:
    adj_matrix (torch.Tensor): The original adjacency matrix.
    rem (float): The tunable threshold.
    device (str): The device on which to perform computations ('cpu' or 'cuda').

    Returns:
    torch.Tensor: The transformed adjacency matrix.
    """
    output = []
    for i in range(len(adj_matrix)):
        # Generate a random matrix M with the same shape as adj_matrix and values in [0, 1)
        M = torch.rand(adj_matrix[i].shape, device=device)
        
        # Create a mask for the diagonal elements
        diagonal_mask = torch.eye(adj_matrix[i].size(0), device=device).bool()
        
        # Apply the transformation condition, but keep the diagonal unchanged
        A_prime = torch.where(M >= rem, adj_matrix[i], torch.zeros_like(adj_matrix[i]))
        
        # Use the diagonal mask to keep the diagonal elements of adj_matrix in A_prime
        A_prime[diagonal_mask] = adj_matrix[i][diagonal_mask]
        output.append(A_prime)
    # print('edge_masking output', output)
    return output
