import numpy as np
import torch
from scipy.stats import wishart



def generate_time_series(n_steps, n_dim, scale_factor=0.9, wishart_df=None, seed=None, value_range=(0, 1), correlation_strength=0.001):
    """
    Generate a time series using a stable random walk matrix and Wishart-distributed noise covariance,
    with values scaled to a specified interval and controllable correlation between components.
    
    Parameters:
    n_steps (int): Number of time steps in the time series.
    n_dim (int or tuple): Dimension of the state vector and the transition matrix. If a tuple is provided,
                          the time series will be reshaped to this shape.
    scale_factor (float): Scaling factor for the transition matrix eigenvalue normalization.
    wishart_df (int, optional): Degrees of freedom for the Wishart distribution. Default is the product of dimensions.
    seed (int, optional): Random seed for reproducibility.
    value_range (tuple, optional): Tuple (min, max) specifying the range to which the time series values should be scaled.
    correlation_strength (float): Controls the correlation between the elements in the time series. Should be between 0 and 1.

    Returns:
    torch.Tensor: The generated time series as a PyTorch tensor with shape (*n_dim, n_steps).
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine the flat dimension size
    if isinstance(n_dim, tuple):
        flat_dim = np.prod(n_dim)
    else:
        flat_dim = n_dim

    # Step 1: Generate the transition matrix A
    A = np.random.randn(flat_dim, flat_dim)
    
    # Introduce correlation into the transition matrix by mixing with an identity matrix
    A = correlation_strength * A + (1 - correlation_strength) * np.identity(flat_dim)
    
    max_eigenvalue = np.max(np.abs(np.linalg.eigvals(A)))
    A = scale_factor * A / max_eigenvalue

    # Step 2: Generate the covariance matrix for epsilon_t
    if wishart_df is None:
        wishart_df = flat_dim  # default degrees of freedom
    
    # Create a correlation matrix and use it to adjust the covariance matrix
    base_cov = wishart.rvs(df=wishart_df, scale=np.identity(flat_dim))
    cov_matrix = correlation_strength * base_cov + (1 - correlation_strength) * np.diag(np.diag(base_cov))

    # Step 3: Generate the time series
    X = np.zeros((n_steps, flat_dim))
    for t in range(1, n_steps):
        epsilon_t = np.random.multivariate_normal(np.zeros(flat_dim), cov_matrix)
        X[t] = np.dot(A, X[t-1]) + epsilon_t

    # Step 4: Scale the time series to the specified value range
    min_val, max_val = value_range
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = min_val + (X - X_min) * (max_val - min_val) / (X_max - X_min)
    
    # Step 5: Reshape the output to the desired shape with n_steps as the last dimension
    if isinstance(n_dim, tuple):
        X_scaled = X_scaled.T.reshape(*n_dim, n_steps)
    else:
        X_scaled = X_scaled.T  # If n_dim is an int, just transpose the array
    
    # Convert the result to a PyTorch tensor
    return torch.tensor(X_scaled, dtype=torch.float32)
