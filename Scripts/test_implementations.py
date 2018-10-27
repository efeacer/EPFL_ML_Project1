import numpy as np 
from implementations import least_squares_GD
from implementations import least_squares_SGD
from implementations import least_squares
from implementations import ridge_regression
from implementations import logistic_regression
from implementations import reg_logistic_regression
from proj1_helpers import load_csv_data
from data_processing import train_test_split
from data_processing import report_prediction_accuracy
from data_processing import report_prediction_accuracy_logistic
from data_processing import standardize
from data_processing import change_labels_logistic

def main():
    """
    Tests the six mandatory implementations on the raw data sets. Splits the
    original training set into a new training set and a test set with the ratio
    of the new training set to the old one being 0.8. Reports the percentage of 
    correct predictions for each method. As a side note, standardization of the 
    data helps algorithms that use gradient descent, hence standardized features
    are used in those iterative algorithms.
    """
    y, tx, _ = load_csv_data('train.csv') 
    y_train, tx_train, y_test, tx_test = train_test_split(y, tx, 0.8)
    standardized_tx_train, _, _= standardize(tx_train)
    standardized_tx_test, _, _ = standardize(tx_test)
    test_least_squares_GD(y_train, standardized_tx_train, y_test, standardized_tx_test)
    test_least_squares_SGD(y_train, standardized_tx_train, y_test, standardized_tx_test)
    test_least_squares(y_train, tx_train, y_test, tx_test)
    test_ridge_regression(y_train, tx_train, y_test, tx_test)
    y_train = change_labels_logistic(y_train)
    y_test = change_labels_logistic(y_test)
    test_logistic_regression(y_train, standardized_tx_train, y_test, standardized_tx_test)
    test_reg_logistic_regression(y_train, standardized_tx_train, y_test, standardized_tx_test)

def test_least_squares_GD(y_train, tx_train, y_test, tx_test):
    """
    Tests least_squares_GD method on the splitted data set and 
    reports percentage of correct predictions. 
    Args:
        y_train: training labels after the splitting
        tx_train: training features after the splitting
        y_test: test labels after the splitting
        tx_test: test features after the splitting
    """
    print('\nTesting least_squares_GD...')
    w, _ = least_squares_GD(y_train, tx_train, np.zeros(tx_train.shape[1]), 1000, 0.05)
    report_prediction_accuracy(y_test, tx_test, w)
    print('... testing completed.')

def test_least_squares_SGD(y_train, tx_train, y_test, tx_test):
    """
    Tests least_squares_SGD method on the splitted data set and 
    reports percentage of correct predictions.
    Args:
        y_train: training labels after the splitting
        tx_train: training features after the splitting
        y_test: test labels after the splitting
        tx_test: test features after the splitting
    """
    print('\nTesting least_squares_SGD...')
    w, _ = least_squares_SGD(y_train, tx_train, np.zeros(tx_train.shape[1]), 3000, 0.005)
    report_prediction_accuracy(y_test, tx_test, w)
    print('... testing completed.')

def test_least_squares(y_train, tx_train, y_test, tx_test):
    """
    Tests least_squares method on the splitted data set and 
    reports percentage of correct predictions. 
    Args:
        y_train: training labels after the splitting
        tx_train: training features after the splitting
        y_test: test labels after the splitting
        tx_test: test features after the splitting
    """
    print('\nTesting least_squares...')
    w, _ = least_squares(y_train, tx_train)
    report_prediction_accuracy(y_test, tx_test, w)
    print('... testing completed.')

def test_ridge_regression(y_train, tx_train, y_test, tx_test):
    """
    Tests ridge_regression method on the splitted data set and 
    reports percentage of correct predictions.
    Args:
        y_train: training labels after the splitting
        tx_train: training features after the splitting
        y_test: test labels after the splitting
        tx_test: test features after the splitting
    """
    print('\nTesting ridge_regression...')
    w, _ = ridge_regression(y_train, tx_train, 1e-08)
    report_prediction_accuracy(y_test, tx_test, w)
    print('... testing completed.')

def test_logistic_regression(y_train, tx_train, y_test, tx_test):
    """
    Tests logistic_regression method on the splitted data set and 
    reports percentage of correct predictions.
    Args:
        y_train: training labels after the splitting
        tx_train: training features after the splitting
        y_test: test labels after the splitting
        tx_test: test features after the splitting
    """
    print('\nTesting logistic_regression...')
    w, _ = logistic_regression(y_train, tx_train, np.zeros(tx_train.shape[1]), 3000, 1e-06)
    report_prediction_accuracy_logistic(y_test, tx_test, w)
    print('... testing completed.')

def test_reg_logistic_regression(y_train, tx_train, y_test, tx_test):
    """
    Tests reg_logistic_regression method on the splitted data set and 
    reports percentage of correct predictions.
    Args:
        y_train: training labels after the splitting
        tx_train: training features after the splitting
        y_test: test labels after the splitting
        tx_test: test features after the splitting
    """
    print('\nTesting reg_logistic_regression...')
    w, _ = reg_logistic_regression(y_train, tx_train, 1, np.zeros(tx_train.shape[1]), 3000, 1e-06)
    report_prediction_accuracy_logistic(y_test, tx_test, w)
    print('... testing completed.')

if __name__ == "__main__":
    main()