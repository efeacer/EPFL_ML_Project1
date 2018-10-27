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

def main():
    y, tx, _ = load_csv_data('train.csv')
    tx, _, _ = standardize(tx)
    y_train, tx_train, y_test, tx_test = train_test_split(y, tx, 0.8)
    test_least_squares_GD(y_train, tx_train, y_test, tx_test)
    test_least_squares_SGD(y_train, tx_train, y_test, tx_test)
    test_least_squares(y_train, tx_train, y_test, tx_test)
    test_ridge_regression(y_train, tx_train, y_test, tx_test)
    test_logistic_regression(y_train, tx_train, y_test, tx_test)
    test_reg_logistic_regression(y_train, tx_train, y_test, tx_test)
    
def test_least_squares_GD(y_train, tx_train, y_test, tx_test):
    print('\nTesting least_squares_GD...')
    w, _ = least_squares_GD(y_train, tx_train, np.zeros(tx_train.shape[1]), 300, 0.1)
    report_prediction_accuracy(y_test, tx_test, w)
    print('... testing completed.')

def test_least_squares_SGD(y_train, tx_train, y_test, tx_test):
    print('\nTesting least_squares_SGD...')
    w, _ = least_squares_GD(y_train, tx_train, np.zeros(tx_train.shape[1]), 300, 0.001)
    report_prediction_accuracy(y_test, tx_test, w)
    print('... testing completed.')

def test_least_squares(y_train, tx_train, y_test, tx_test):
    print('\nTesting least_squares...')
    w, _ = least_squares(y_train, tx_train)
    report_prediction_accuracy(y_test, tx_test, w)
    print('... testing completed.')

def test_ridge_regression(y_train, tx_train, y_test, tx_test):
    print('\nTesting ridge_regression...')
    w, _ = ridge_regression(y_train, tx_train, 1e-08)
    report_prediction_accuracy(y_test, tx_test, w)
    print('... testing completed.')

def test_logistic_regression(y_train, tx_train, y_test, tx_test):
    print('\nTesting logistic_regression...')
    w, _ = logistic_regression(y_train, tx_train, np.zeros(tx_train.shape[1]), 2000, 1e-06)
    report_prediction_accuracy_logistic(y_test, tx_test, w)
    print('... testing completed.')

def test_reg_logistic_regression(y_train, tx_train, y_test, tx_test):
    print('\nTesting reg_logistic_regression...')
    w, _ = reg_logistic_regression(y_train, tx_train, 1, np.zeros(tx_train.shape[1]), 2000, 1e-06)
    report_prediction_accuracy_logistic(y_test, tx_test, w)
    print('... testing completed.')

if __name__ == "__main__":
    main()