# Necessary import(s)
import numpy as np
from helper_functions import compute_error_vector
from helper_functions import compute_gradient
from helper_functions import compute_mse
from helper_functions import compute_logistic_gradient
from helper_functions import compute_logistic_loss
from helper_functions import penalized_logistic_regression

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
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        error_vector = compute_error_vector(y, tx, w)
        loss = compute_mse(error_vector)
        gradient_vector = compute_gradient(tx, error_vector)
        w = w - gamma * gradient_vector
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]

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
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        random_index = np.random.randint(len(y))
        # sample a random data point from y vector
        y_random = y[random_index] 
        # sample a random row vector from tx matrix
        tx_random = tx[random_index] 
        error_vector = compute_error_vector(y_random, tx_random, w)
        loss = compute_mse(error_vector)
        gradient_vector = compute_gradient(tx_random, error_vector)
        w = w - gamma * gradient_vector
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Args:
        y: labels
        tx: features
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
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
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    coefficient_matrix = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    error_vector = compute_error_vector(y, tx, w)
    loss = compute_mse(error_vector)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using stochastic gradient descent 
    Args:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on logistic loss
    """
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        gradient_vector = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient_vector
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent 
    Args:
        y: labels
        tx: features
        lambda_: regularization parameter
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on logistic loss
    """
    threshold = 1e-8
    w = initial_w
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        loss, gradient_vector = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient_vector
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]
