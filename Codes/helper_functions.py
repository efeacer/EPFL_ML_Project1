#Necessary import(s)
import numpy as np
import implementations as imp

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

def standardize(x, mean_x = None, std_x = None):
    """
    Standardizes the original data set.

    Args:
        x: data set to standardize
        mean_x: mean of the data set, can be specified or computed
        std_x: standard deviation of the data set, can be specified or computed

    Returns:
        x: standardized data set
        mean_x: mean of the data set
        std_x: standard deviation of the data set
    """
    mean_x = mean_x or np.mean(x)
    x = x - mean_x
    std_x = std_x or np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def compute_rmse(loss_mse): 
    """
    Computes the root mean squared error.

    Args:
        loss_mse: numeric value of the mean squared error loss

    Returns:
        loss_rmse: numeric value of the root mean squared error loss
    """
    return np.sqrt(2 * loss_mse)
    
def build_k_indices(y, k_fold, seed):
    """
    Randomly partitions the indices of the data set into k groups

    Args:
        y: labels, used for indexing
        k_fold: number of groups after the partitioning
        seed: the random seed value

    Returns:
        k_indices: an array of k sub-indices that are randomly partitioned
    """
    num_rows = y.shape[0]
    interval = int(num_rows / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_rows)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """
    Performs cross_validation for a specific test set from the partitioned set.

    Args:
        y: labels
        x: features
        k_indices: an array of k sub-indices that are randomly partitioned
        k: the test set that is kth partition 
        lambda_: regularization parameter for the ridge regression
        degree: degree of the polynomial basis for the feature expansion

    Returns:
        (rmse_training, rmse_test): (numeric value of the root mean squared error loss
            for the training set, numeric value of the root mean squared error loss
            for the test set)
    """
    y_test = y[k_indices[k]]
    y_training = np.delete(y, k_indices[k])
    x_test = x[k_indices[k]]
    x_training = np.delete(x, k_indices[k], axis = 0)
    augmented_x_test = build_polynomial(x_test, degree)
    augmented_x_training = build_polynomial(x_training, degree)
    w, loss_training = imp.ridge_regression(y_training, augmented_x_training, lambda_)
    loss_test = compute_mse(compute_error_vector(y_test, augmented_x_test, w))
    return compute_rmse(loss_training), compute_rmse(loss_test)

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
    return np.sum(np.log(1. + np.exp(tx.dot(w))) - y * tx.dot(w))

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
