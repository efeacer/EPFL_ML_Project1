# Necessary import(s)
import numpy as np

# The helper methods used by the learning methods above are implemented below:

def compute_error_vector(y, tx, w):
    """
    Computes the error vector that is defined as y - tx . w
    Args:
        y: labels 
        tx: features
        w: weight vector
    Returns:
        error_vector: the error vector defined as y - tx.dot(w)
    """
    return y - tx.dot(w)

def compute_mse(error_vector):
    """
    Computes the mean squared error for a given error vector.
    Args:
        error_vector: error vector computed for a specific dataset and model
    Returns:
        mse: numeric value of the mean squared error
    """
    return np.mean(error_vector ** 2) / 2

def compute_gradient(tx, error_vector):
    """
    Computes the gradient for the mean squared error loss function.
    Args:
        y: labels
        error_vector: error vector computed for a specific data set and model
    Returns:
        gradient: the gradient vector computed according to its definition
    """
    return - tx.T.dot(error_vector) / error_vector.size

def build_polynomial(x, degree):
    """
    Extends the feature matrix, x, by adding a polynomial basis of the given degree.
    Args:
        x: features
        degree: degree of the polynomial basis
    Returns:
        augmented_x: expanded features based on a polynomial basis
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

def compute_rmse(loss_mse): 
    """
    Computes the root mean squared error.
    Args:
        loss_mse: numeric value of the mean squared error loss
    Returns:
        loss_rmse: numeric value of the root mean squared error loss
    """
    return np.sqrt(2 * loss_mse)
    
def sigmoid(t):
    """
    Applies the sigmoid function to a given input t.
    Args:
        t: the given input to which the sigmoid function will be applied.
    Returns:
        sigmoid_t: the value of sigmoid function applied to t
    """
    return 1. / (1. + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """
    Computes the loss as the negative log likelihood of picking the correct label.
    Args:
        y: labels 
        tx: features
        w: weight vector
    Returns:
        loss: the negative log likelihood of picking the correct label
    """
    tx_dot_w = tx.dot(w)
    return np.sum(np.log(1. + np.exp(tx_dot_w)) - y * tx_dot_w)

def compute_logistic_gradient(y, tx, w):
    """
    Computes the gradient of the loss function used in logistic regression.
    Args:
        y: labels 
        tx: features
        w: weight vector
    Returns:
        logistic_gradient: the gradient of the loss function used in 
            logistic regression.
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y)

def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Adds the penalization term (2-norm of w vector) on top of the normal
    logistic loss. Computes the modified loss and gradient.
    Args:
        y: labels 
        tx: features
        w: weight vector
    Returns:
        loss: the modified version of the normal logistic loss
        logistic_gradient: the gradient of modified loss function used in 
            penalized logistic regression.
    """
    loss = compute_logistic_loss(y, tx, w) + (lambda_ / 2) * w.T.dot(w)
    gradient = compute_logistic_gradient(y, tx, w) + lambda_ * w
    return loss, gradient

def cross_terms(x, x_initial):
    """
    Adds the multiplication of different features as new features.
    Args:
        x: the given feature matrix
        x_initial: the features whose multiplications will be added
    Returns:
        x_cross_terms: feature matrix with cross terms
    """
    for col1 in range(x_initial.shape[1]):
        for col2 in np.arange(col1 + 1, x_initial.shape[1]):
            if col1 != col2:
                x = np.c_[x, x_initial[:, col1] * x_initial[:, col2]]
    return x

def log_terms(x, x_initial):
    """
    Adds the logarithms of features as new features.
    Args:
        x: the given feature matrix
        x_initial: the features whose logarithms will be added
    Returns:
        x_log_terms: feature matrix with logarithms
    """
    for col in range(x_initial.shape[1]):
        current_col = x_initial[:, col]
        current_col[current_col <= 0] = 1
        x = np.c_[x, np.log(current_col)]
    return x

def sqrt_terms(x, x_initial):
    """
    Adds the square roots of features as new features.
    Args:
        x: the given feature matrix
        x_initial: the features whose square roots will be added
    Returns:
        x_sqrt_terms: feature matrix with square roots
    """
    for col in range(x_initial.shape[1]):
        current_col = np.abs(x_initial[:, col])
        x = np.c_[x, np.sqrt(current_col)]
    return x

def apply_trigonometry(x, x_initial):
    """
    Adds the sin and cos of features as new features.
    Args:
        x: the given feature matrix
        x_initial: the features whose sin and cos will be added
    Returns:
        x_sqrt_terms: feature matrix with sine values
    """
    for col in range(x_initial.shape[1]):
        x = np.c_[x, np.sin(x_initial[:, col])]
        x = np.c_[x, np.cos(x_initial[:, col])]
    return x

def feature_engineering(x, degree, has_angles = False):
    """
    Builds a polynomial with the given degree from the initial features,
    add the cross terms, logarithms and square roots of the initial features
    as new features. Also includes the sine of features as an option.
    Args:
        x: features
        degree: degree of the polynomial basis
        has_angles: Boolean value to determine including sin and cos of features
    Returns:
        x_engineered: engineered features
    """
    x_initial = x
    x = build_polynomial(x, degree)
    x = cross_terms(x, x_initial)
    x = log_terms(x, x_initial)
    x = sqrt_terms(x, x_initial)
    if has_angles:
        x = apply_trigonometry(x, x_initial)
    return x