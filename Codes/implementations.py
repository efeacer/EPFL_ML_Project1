import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    Args:
        y:
        tx:
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size

    Returns:
        (w, loss):
    """
    w = initial_w
    for _ in range(max_iters):
        error_vector = compute_error_vector(y, w, tx)
        gradient_vector = compute_gradient(tx, error_vector)
        w = w - gamma * gradient_vector
    final_error_vector = compute_error_vector(y, tx, w)
    loss = compute_mse(final_error_vector)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent

    Args:
        y:
        tx:
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size

    Returns:
        (w, loss):
    """
    w = initial_w
    for _ in range(max_iters):
        # sample a random data point from y vector
        y_random = y[np.random.randint(len(y))] 
        # sample a random row vector from tx matrix
        tx_random = tx[np.random.randint(len(tx))] 
        error_vector = compute_error_vector(y_random, tx_random, w)
        stochastic_gradient_vector = compute_gradient(tx_random, error_vector)
        w = w - gamma * stochastic_gradient_vector
    final_error_vector = compute_error_vector(y, tx, w)
    loss = compute_mse(final_error_vector)
    return w, loss

def least_squares(y, tx):
    """
    Least squares regression using normal equations

    Args:
        y:
        tx:

    Returns:
        (w, loss):
    """
    coefficient_matrix = tx.T.dot(tx)
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    error_vector = compute_error_vector(y, tx, w)
    loss = compute_mse(error_vector)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

    Args:
        y:
        tx:
        lambda_: regularization parameter

    Returns:
        (w, loss):
    """
    coefficient_matrix = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    error_vector = compute_error_vector(y, tx, w)
    loss = compute_mse(error_vector)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD

    Args:
        y:
        tx:
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size

    Returns:
        (w, loss):
    """
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD

    Args:
        y:
        tx:
        lambda_: regularization parameter
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size

    Returns:
        (w, loss):
    """
    raise NotImplementedError

def compute_error_vector(y, tx, w):
    """
    Computes the error vector that is defined as y - tx . w

    Args:
        y:
        tx:
        w:

    Returns:
        error_vector:
    """
    return y - tx.dot(w)

def compute_mse(error_vector):
    """
    Computes the mean squared error for a given error vector.

    Args:
        error_vector:

    Returns:
        mse:
    """
    return np.mean(error_vector ** 2) / 2

def compute_gradient(tx, error_vector):
    """
    Computes the gradient for the mean squared error loss function.

    Args:
        y:
        error_vector:

    Returns:
        gradient:
    """
    return - tx.T.dot(error_vector) / len(error_vector)