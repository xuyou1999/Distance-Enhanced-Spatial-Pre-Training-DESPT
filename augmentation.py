import numpy as np
import torch
from scipy.fftpack import dct, idct

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

def temporal_shifting(input_tensor, r=0.5):
    """
    Apply temporal shifting to a 2D tensor according to the provided formula.
    
    Parameters:
    - input_tensor: a 2D tensor of shape [n, t], where n is the number of instances, and t is the timestamps.
    - r: the hyperparameter to control the distribution U(r, 1) for alpha.

    Returns:
    - The augmented tensor after applying temporal shifting.
    """
    n, t = input_tensor.shape
    # Ensure there are enough timestamps for shifting
    if t < 2:
        raise ValueError("The tensor does not have enough timestamps for shifting.")
    
    # Generate alpha from uniform distribution U(r, 1)
    alpha = np.random.uniform(r, 1)
    # print('alpha', alpha)
    
    # Apply temporal shifting according to the provided formula
    shifted_tensor = alpha * input_tensor[:, :(t-1)] + (1 - alpha) * input_tensor[:, 1:]
    
    # To maintain the original shape, the last column is replicated from the second last column of the shifted tensor
    # This assumes the intention is to 'forward-fill' the last timestamp, as we can't generate new future information
    last_column = shifted_tensor[:, -1].unsqueeze(1)
    shifted_tensor = torch.cat((shifted_tensor, last_column), dim=1)
    
    return shifted_tensor


def temporal_shifting_new(input_tensor, r=0.5):
    """
    Apply temporal shifting to a 2D tensor according to the provided formula.
    
    Parameters:
    - input_tensor: a 2D tensor of shape [n, t], where n is the number of instances, and t is the timestamps.
    - r: the hyperparameter to control the distribution U(r, 1) for alpha.

    Returns:
    - The augmented tensor after applying temporal shifting.
    """
    n, t = input_tensor.shape
    # Ensure there are enough timestamps for shifting
    if t < 2:
        raise ValueError("The tensor does not have enough timestamps for shifting.")
    
    # Generate alpha from uniform distribution U(r, 1)
    alphas = np.random.uniform(r, 1, size=n)
    alphas = torch.from_numpy(alphas).float().to(input_tensor.device)

    # Apply temporal shifting according to the provided formula, using a unique alpha for each input
    shifted_tensor = alphas[:, None] * input_tensor[:, :-1] + (1 - alphas[:, None]) * input_tensor[:, 1:]
    
    # To maintain the original shape, the last column is replicated from the second last column of the shifted tensor
    # This assumes the intention is to 'forward-fill' the last timestamp, as we can't generate new future information
    last_column = shifted_tensor[:, -1].unsqueeze(1)
    shifted_tensor = torch.cat((shifted_tensor, last_column), dim=1)
    
    return shifted_tensor

def input_smoothing(data, r, e):
    """
    Perform input smoothing on the data tensor without using an adjacency matrix.
    
    Parameters:
        data (torch.Tensor): The input data tensor of shape (num_sensors, data_length).
        r (float): Scaling factor for random matrix generation.
        e (int): Number of low-frequency components to keep unchanged.
        
    Returns:
        torch.Tensor: Smoothed data tensor.
    """
    num_sensors, data_length = data.shape
    
    # Convert data from PyTorch tensor to NumPy array
    data_np = data.cpu().numpy()
    
    # Apply DCT to each node's data
    X_dct = dct(data_np, axis=1, norm='ortho')
    
    # Generate random matrix M
    L_minus_Eis = data_length - e
    M = np.random.uniform(r, 1, (num_sensors, L_minus_Eis))
    
    # Scale high-frequency components
    X_dct[:, e:] *= M
    
    # Apply inverse DCT to convert back to time domain
    X_idct = idct(X_dct, axis=1, norm='ortho')
    
    # Convert the result back to PyTorch tensor
    smoothed_data = torch.tensor(X_idct, dtype=data.dtype)
    
    return smoothed_data