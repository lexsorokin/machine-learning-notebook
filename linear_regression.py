import numpy as np

def linear_model(w, x, b):
    """
    Computes y_hat for linear regression

    For LR with one variable w & x are scalar
    For LR with multiple variables w & x are vectors
    """ 
    return np.dot(w, x) + b


def compute_cost(x, y, w, b, lambda_ = 1):
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
    n = len(w)

    total_cost = 0 

    for i in range(m): 
        y_hat = linear_model(w, x[i], b)
        squared_err = (y_hat - y[i]) ** 2
        total_cost += squared_err
    
    total_cost = (1 / (2 * m)) * total_cost

    reg_cost = 0

    for j in range(n):
        reg_cost += w[j] ** 2
    
    reg_cost = (lambda_ / (2 * m)) * reg_cost

    total_cost = total_cost + reg_cost

    return total_cost


def compute_cost_vectorized(x, y, w, b, lambda_ = 1): 
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

    y_hat = x @ w + b

    error = y_hat - y

    mse_cost = (1 / (2 * m)) * np.sum(error ** 2)

    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)

    total_cost = mse_cost + reg_cost

    return total_cost


def compute_gradient(x, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = x.shape

    dj_dw = np.zeros_like(w)
    dj_db = 0

    for i in range(m): 
        y_hat = linear_model(w, x[i], b)
        err = y_hat - y[i]
        for j in range(n): 
            dj_dw[j] += err * x[i]
        dj_db += err

    dj_dw = (1 / m) * dj_dw
    dj_db = (1 / m) * dj_db

    for j in range(n): 
        dj_dw[j] += (lambda_ / m) * w[j]


    return dj_dw, dj_db


def compute_gradient_vectorized(x, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m = x.shape[0]

    y_hat = x @ w + b  # shape (m,)

    error = y_hat - y  # shape (m,)

    dj_dw = (x.T @ error) / m + (lambda_ / m) * w  # shape (n,)
    dj_db = np.sum(error) / m  # scalar

    return dj_dw, dj_db


def check_convergence(prev_cost, curr_cost, epsilon_=1e-3):
    if prev_cost == 0: 
        return False
    return abs(prev_cost - curr_cost) < epsilon_


def scale(x, method="z_score_normalization", dimension=1): 
    
    if method == "max_normalization": 
        return _max_normalize(x, dimension)
    if method == "mean_normalization": 
        return _mean_normalization(x, dimension)
    if method == "z_score_normalization": 
        return _z_score_normalization(x, dimension)
    
    return Exception(f"Provided normalization method {method} is not valid")

def normalize_y(y): 
    y_mean = np.mean(y)
    y_std = np.std(y)
    return (y - y_mean) / (y_std if y_std != 0 else 1), y_mean, y_std

def denormalize_y(y_pred_normalized, mean, std): 
    return y_pred_normalized * std + mean
        
def _max_normalize(x):
    """
    x: ndarray (m, n) - m примеров, n фич
    """
    max_per_feature = np.max(x, axis=0)  # максимум по каждому столбцу
    x_normalized = x / max_per_feature    # делим каждый столбец на его максимум
    return x_normalized


def _mean_normalization(x): 
    """
    x: ndarray (m, n) - m примеров, n фич
    """
    max_per_feature = np.max(x, axis=0)
    min_per_feature = np.min(x, axis=0)
    mean_per_feature = np.mean(x, axis=0)
    x_normalized = (x - mean_per_feature) / (max_per_feature - min_per_feature)

    return x_normalized
        

def _z_score_normalization(x):
    """
    x: ndarray (m, n) - m примеров, n фич
    """
    mean_per_feature = np.mean(x, axis=0)
    std_per_feature = np.std(x, axis=0)
    
    # защита от деления на 0 (если фича константная)
    std_per_feature[std_per_feature == 0] = 1
    
    x_normalized = (x - mean_per_feature) / std_per_feature
    return x_normalized


def fit(x, y, iterations=1000, alpha_=1, lambda_=1, epsilon_=1e-3):

    n = x.shape[1] 

    w = np.zeros(n)
    b = 0
  
    prev_Jwb = 0

    for _ in range(iterations): 
        Jwb = compute_cost_vectorized(x, y, w, b, lambda_)
        has_converged = check_convergence(prev_Jwb, Jwb, epsilon_=epsilon_)
        if has_converged: 
            break
        
        prev_Jwb = Jwb
        
        dj_dw, dj_db = compute_gradient_vectorized(x, y, w, b, lambda_)
        tmp_w = w - (alpha_*dj_dw)
        tmp_b = b - (alpha_*dj_db)

        w, b = tmp_w, tmp_b

    return w, b


def train(x, y):
    x_normalized = scale(x)
    y_normalized, y_mean, y_std  = normalize_y(y)

    w, b = fit(x_normalized, y_normalized)

    y_hat = linear_model(w, x_normalized, b)

    y_hat = denormalize_y(y_hat, y_mean, y_std)
    
    return w, b, y_hat 
     


    

        
