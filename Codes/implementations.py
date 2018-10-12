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
    return - tx.T.dot(error_vector) / error_vector.size

def build_polynomial(x, degree):
    """
    Extends the feature matrix, x, by adding a polynomial basis of 
    the given degree.

    Args:
        x: feature matrix
        degree: degree of the polynomial basis

    Returns:
        augmented_x:
    """
    num_cols = x.shape[1] if len(x.shape) > 1 else 1
    augmented_x = np.ones((len(x), 1))
    for col in range(num_cols):
        for degree in range(1, degree + 1):
            if num_cols > 1:
                augmented_x = np.c_[augmented_x, np.power(x[ :, col], degree)]
            else:
                augmented_x = np.c_[augmented_x, np.power(x, degree)]
        if num_cols > 1 and col != num_cols - 1:
            augmented_x = np.c_[augmented_x, np.ones((len(x), 1))]
    return augmented_x

def standardize(x, mean_x = None, std_x = None):
    """
    Standardize the original data set.

    Args:
        x: data set to standardize
        mean_x: 
        std_x:

    Returns:
        x:
        mean_x:
        std_x:
    """
    mean_x = mean_x or np.mean(x)
    x = x - mean_x
    std_x = std_x or np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def build_k_indices(y, k_fold, seed):
    num_rows = y.shape[0]
    interval = int(num_rows / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_rows)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def compute_rmse(loss_mse): 
    return np.sqrt(2 * loss_mse)
    
def cross_validation(y, x, k_indices, k, lambda_, degree):
    y_test = y[k_indices[k]]
    y_training = np.delete(y, k_indices[k])
    x_test = x[k_indices[k]]
    x_training = np.delete(x, k_indices[k], axis = 0)
    augmented_x_test = build_polynomial(x_test, degree)
    augmented_x_training = build_polynomial(x_training, degree)
    w, loss_training = ridge_regression(y_training, augmented_x_training, lambda_)
    loss_test = compute_mse(compute_error_vector(y_test, augmented_x_test, w))
    return compute_rmse(loss_training), compute_rmse(loss_test)
