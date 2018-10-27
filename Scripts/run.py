# Necessary import(s)
import helper_functions as hf
import numpy as np
import sys
import argparse
from data_processing import process_data
from data_processing import report_prediction_accuracy
from data_processing import build_k_indices
from data_processing import cross_validation
from data_processing import standardize
from helper_functions import build_polynomial
from implementations import ridge_regression
from proj1_helpers import load_csv_data
from proj1_helpers import predict_labels
from proj1_helpers import create_csv_submission

training_files = ['train_jet_0.csv', 'train_jet_1.csv', 'train_jet_2.csv', 'train_jet_3.csv']
test_files = ['test_jet_0.csv', 'test_jet_1.csv', 'test_jet_2.csv', 'test_jet_3.csv']

def main(pd, gs):
    if pd:
        process_data('train.csv', 'test.csv')
    y_train_jets = []
    tx_train_jets = []
    ids_train_jets = []
    y_test_jets = []
    tx_test_jets = []
    ids_test_jets = []
    load_data_sets(y_train_jets, tx_train_jets, ids_train_jets, y_test_jets, tx_test_jets, ids_test_jets)
    degree_best_jets = [8, 10, 9, 9]
    lambda_best_jets = [5e-05, 6e-08, 7e-08, 0.005]
    if gs:
        perform_grid_search_with_cross_validation(degree_best_jets, lambda_best_jets, y_train_jets, tx_train_jets)
    predictions = []
    ids_predicted = []
    learn(predictions, ids_predicted, y_train_jets, tx_train_jets, tx_test_jets, ids_test_jets,
    lambda_best_jets, degree_best_jets)
    combine_and_create_submission(predictions, ids_predicted, 'submit_E_M_D')
    
def load_data_sets(y_train_jets, tx_train_jets, ids_train_jets, y_test_jets, tx_test_jets, ids_test_jets):
    print('\nLoading the processed training and test set data for each jet number...')
    for jet_num in range(4):
        y_train, tx_train, ids_train = load_csv_data(training_files[jet_num])
        y_train_jets.append(y_train)
        tx_train_jets.append(tx_train)
        ids_train_jets.append(ids_train)
        y_test, tx_test, ids_test = load_csv_data(test_files[jet_num])
        y_test_jets.append(y_test)
        tx_test_jets.append(tx_test)
        ids_test_jets.append(ids_test)
        print('\nTraining and test set data for jet ', str(jet_num), ' is loaded.')
    print('\n... done.')

def perform_grid_search_with_cross_validation(degree_best_jets, lambda_best_jets, y_train_jets, tx_train_jets):
    k_fold = 5
    degrees = np.arange(7, 13)
    lambda_powers = np.arange(-10, -1)
    lambda_numbers = np.arange(1, 10)
    for jet_num in range(4):
        min_test_rmse = sys.maxsize
        lambda_best = None
        degree_best = None
        print('\nGrid search with cross validation for jet ', str(jet_num), '...')
        y_train, tx_train = y_train_jets[jet_num], tx_train_jets[jet_num]
        k_indices = build_k_indices(y_train, k_fold, seed = 1)
        for degree in degrees:
            augmented_tx_train = build_polynomial(tx_train, degree)
            for lambda_power in lambda_powers:
                for lambda_number in lambda_numbers:
                    lambda_ = lambda_number * (10. ** lambda_power)
                    for k in range(k_fold):
                        _, test_rmse = cross_validation(y_train, augmented_tx_train, k_indices, k, lambda_)
                        if test_rmse < min_test_rmse:
                            min_test_rmse = test_rmse
                            lambda_best = lambda_
                            degree_best = degree
        print('\nOptimal hyperparameters:')
        print('- polynomial degree = ', degree_best)
        degree_best_jets[jet_num] = degree_best
        print('- lambda = ', lambda_best)
        lambda_best_jets[jet_num] = lambda_best
        print('\n- minimum test set rmse = ', min_test_rmse)
        print('\n... finished')

def learn(predictions, ids_predicted, y_train_jets, tx_train_jets, tx_test_jets, ids_test_jets,
lambda_best_jets, degree_best_jets):
    print('\nLearning by ridge regression...')
    for jet_num in range(4):
        print('\nLearning from training set with jet number ', str(jet_num), ' using optimal hyperparameters...')
        y_train, tx_train = y_train_jets[jet_num], tx_train_jets[jet_num]
        tx_train = build_polynomial(tx_train, degree_best_jets[jet_num])
        w_best, _ = ridge_regression(y_train, tx_train, lambda_best_jets[jet_num])
        tx_test, ids_test = tx_test_jets[jet_num], ids_test_jets[jet_num]
        tx_test = build_polynomial(tx_test, degree_best_jets[jet_num])
        predictions.append(predict_labels(w_best, tx_test))
        ids_predicted.append(ids_test)
        print('\nReporting prediction accuracy found in the 5 folded cross validation... \n')
        report_predictions(lambda_best_jets[jet_num], y_train, tx_train)
        print('\n... this gives a rough idea about the real submission score.')
        print('\n... predicted labels for test set with jet number ', str(jet_num))
    print('\n... ,predicted labels for each test set.')

def report_predictions(lambda_best, y_train, augmented_tx_train):
    k_fold = 5
    k_indices = build_k_indices(y_train, k_fold, seed = 1)
    for k in range(k_fold):
        cross_validation(y_train, augmented_tx_train, k_indices, k, lambda_best, report_predictions = True)

def combine_and_create_submission(predictions, ids_predicted, submission_name):
    ids_gathered = []
    predictions_gathered = []
    current_id = min(ids_predicted[:][0])
    length = np.sum(len(prediction) for prediction in predictions)
    print('\nGathering ids and predictions for each jet number together...')
    for _ in range(length):
        for jet_num in range(4):
            if len(ids_predicted[jet_num]) > 0:
                if ids_predicted[jet_num][0] == current_id:
                    predictions_gathered.append(predictions[jet_num][0])
                    ids_gathered.append(current_id)
                    predictions[jet_num] = np.delete(predictions[jet_num], 0)
                    ids_predicted[jet_num] = np.delete(ids_predicted[jet_num], 0)
                    break
        current_id += 1
    print('\n... ids and predictions for each jet number were gathered.')
    print('\n Creating submission file with name ', str(submission_name), ' ...')
    create_csv_submission(np.array(ids_gathered), np.array(predictions_gathered), submission_name)
    print('\n... ', str(submission_name), ' is created. Ready to submit :) !' )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Runs the procedure to obtain group E_M_D\'s best submission')
    parser.add_argument('-pd', action = 'store_true', help = 'Splits, analyses and modifies raw data', 
    default = False)
    parser.add_argument('-gs', action = 'store_true',
                        help = 'Tunes the hyperparameters using grid search with cross validation. (Hard coded' + 
                        ' optimal hyperparameters will be used if this argument is omitted)',
                        default = False)
    args = parser.parse_args()
    main(args.pd, args.gs)



