import numpy as np

def linear_model(w, x, b):
    """
    Computes y_hat for linear regression

    For LR with one variable w & x are scalar
    For LR with multiple variables w & x are vectors
    """ 
    return np.dot(w, x) + b


def compute_cost(x, y, w, b, lambda_ = 0):
    """
    Computes the cost over all examples (squared error)
    Args:
      x (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m = x.shape[0]

    total_cost = 0 

    for i in range(m): 
        y_hat = linear_model(w, x[i], b)
        squared_error = (y_hat - y[i]) ** 2
        total_cost += squared_error
    
    total_cost = (1 / (2 * m)) * total_cost

    return total_cost


