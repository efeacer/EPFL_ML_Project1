#Necessary import(s)
import numpy as np
from helper_functions import compute_error_vector
from helper_functions import compute_gradient
from helper_functions import compute_mse
from helper_functions import compute_logistic_gradient
from helper_functions import compute_logistic_loss


# The six compulsory learning methods are as implemented as follows:

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    Args:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size

    Returns:
        (w, loss): (optimized weight vector for the model, 
            optimized final loss based on mean squared error)
    """
    w = initial_w
    for _ in range(max_iters):
        error_vector = compute_error_vector(y, tx, w)
        gradient_vector = compute_gradient(tx, error_vector)
        w = w - gamma * gradient_vector
    final_error_vector = compute_error_vector(y, tx, w)
    loss = compute_mse(final_error_vector)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent

    Args:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size

    Returns:
        (w, loss): (optimized weight vector for the model, 
            optimized final loss based on mean squared error)
    """
    w = initial_w
    for _ in range(max_iters):
        random_index = np.random.randint(len(y))
        # sample a random data point from y vector
        y_random = y[random_index] 
        # sample a random row vector from tx matrix
        tx_random = tx[random_index] 
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
        y: labels
        tx: features

    Returns:
        (w, loss): (optimized weight vector for the model, 
            optimized final loss based on mean squared error)
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
        y: labels
        tx: features
        lambda_: regularization parameter

    Returns:
        (w, loss): (optimized weight vector for the model, 
            optimized final loss based on mean squared error)
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
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size

    Returns:
        (w, loss): (optimized weight vector for the model, 
            optimized final loss based on mean squared error)
    """
    w = initial_w
    for _ in range(max_iters):
        random_index = np.random.randint(len(y))
        # sample a random data point from y vector
        y_random = y[random_index] 
        # sample a random row vector from tx matrix
        tx_random = tx[random_index] 
        stochastic_gradient_vector = compute_logistic_gradient(y_random, tx_random, w)
        w = w - gamma * stochastic_gradient_vector
    loss = compute_logistic_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD

    Args:
        y: labels
        tx: features
        lambda_: regularization parameter
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size

    Returns:
        (w, loss): (optimized weight vector for the model, 
            optimized final loss based on mean squared error)
    """
    raise NotImplementedError
