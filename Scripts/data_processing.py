# Necessary import(s)
import csv
import numpy as np
from proj1_helpers import load_csv_data
from implementations import ridge_regression
from helper_functions import compute_error_vector
from helper_functions import compute_mse
from helper_functions import compute_rmse

def load(trainFile, testFile):
    """
    Builds various numpy arrays from the given .csv format training 
    and test tests.
    Args:
        trainFile: file name/path for the input training set
        testFile: file name/path for the input test set
    Returns: 
        y_train: labels in the training set as a numpy array
        tx_train: features in the training set as a numpy array
        ids_train: ids of the training data points as a numpy array
        y_test: labels in the test set as a numpy array
        tx_test: features in the test set as a numpy array
        ids_test: ids of the test data points as a numpy array
    """
    print('\nLoading the raw training and test set data...')
    y_train, tx_train, ids_train = load_csv_data(trainFile)
    y_test, tx_test, ids_test = load_csv_data(testFile)
    print('\n... finished.')
    return y_train, tx_train, ids_train, y_test, tx_test, ids_test

def get_header(file):
    """
    Captures the header line of a given .csv file.
    Args:
        file: file name/path for the given .csv file
    Returns:
        header: the header line of the given .csv file.
    """
    read_file = open(file, 'r')
    reader = csv.DictReader(read_file)
    return reader.fieldnames

def analyse(tx):
    """
    Analyses a given set of features. Marks the features with zero
    variance as the features to be deleted from the data set. Replaces
    each instance of a null(-999) valued feature point with the mean 
    of the non null valued feature points.
    Args: 
        tx: the numpy array representing the given set of features
    Returns:
        columns_to_remove: indices of the features with zero variance,
            which will be removed from the numpy array
    """
    num_cols = tx.shape[1]
    print('\nNumber of columns in the data matrix: ', num_cols)
    columns_to_remove = []
    print('Analysis for data:\n')
    for col in range(num_cols):
        current_col = tx[:, col]
        if len(np.unique(current_col)) == 1:
            print('The column with index ', col, ' is all the same, it will be deleted.')
            columns_to_remove.append(col)
        else:
            current_col[current_col == -999] = np.mean(current_col[current_col != -999])
            print('null values in the ', col, ' indexed column are replaced with the mean.')
    return columns_to_remove

def remove_columns(tx, header, columns_to_remove):
    """
    Removes the given features from the given set of features.
    Args:
        tx: the numpy array representing the given set of features
        header: the header line of the .csv representing the data set
        columns_to_remove: The indices of the features that will be removed 
            from the numpy array of features
    """
    print("\nRemoving columns...")
    num_removed = 0
    for col in columns_to_remove:
        tx = np.delete(tx, col - num_removed, 1)
        header = np.delete(header, col - num_removed + 2)
        num_removed += 1
    print("\n... finished.")
    return tx, header

def create_csv(output_file, y, tx, ids, header, is_test):
    """
    Creates a .csv file and formats it such that it carries the given
    data for labels, features and ids.
    Args:
        output_file: name of the created .csv file
        y: a numpy array representing the given labels
        tx: a numpy array representing the given features
        ids: a numpy array representing the ids of the data points
        header: header of the output .csv file
        is_test: a Boolean indicating whether the output .csv file
            is a test file or a training file, the reason for this
            is that the test set should have unknown labels.
    """
    print('\nCreating new csv file named ' + str(output_file) + '...')
    with open(output_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter = ',', fieldnames = header)
        writer.writeheader()
        for id, y_row, tx_row in zip(ids, y, tx):
            if is_test:
                prediction = '?'
            else:
                prediction = 'b' if y_row == -1 else 's'
            dictionary = {'Id': int(id), 'Prediction': prediction}
            for index in range(len(tx_row)):
                dictionary[header[index + 2]] = float(tx_row[index])
            writer.writerow(dictionary)
        print('\n... finished.')

def split_data(y, tx, ids, jet_num):
    """
    Splits the given data set such that only the data points with a certain 
    jet number remains, where jet number is a discrete valued feature. In
    other words, filters the data set using the jet number.
    Args: 
        y: a numpy array representing the given labels
        tx: a numpy array representing the given features
        ids: a numpy array representing the ids of the data points
        jet_num: the certain value of the discrete feature jet number
    Returns:
        y_masked: numpy array of labels of data points having the specified jet number
        tx_masked: numpy array of features of data points having the specified jet number
        ids_masked: numpy array of ids of data points having the specified jet number
    """
    mask = tx[:, 22] == jet_num
    return y[mask], tx[mask], ids[mask]

def process_data(trainFile, testFile):
    """
    Builds 4 new training data set files and 4 new test training data set files.
    First, splits the initial data tests using the discrete valued feature jet number,
    which can only take the values 0, 1, 2 and 3. Second, processes the splitted data 
    sets by replacing null values and deteling zero varience features. 
    Args:
        trainFile: file name/path for the input training set
        testFile: file name/path for the input test set
    """
    y_train, tx_train, ids_train, y_test, tx_test, ids_test = load(trainFile, testFile)
    header_train = get_header(trainFile)
    header_test = get_header(testFile)
    print('\nData set will be splitted into four, each representing data with different jet numbers.')
    for jet_num in range(4):
        print('\nProcessing training set with jet number = ' + str(jet_num) + '...')
        y_train_jet, tx_train_jet, ids_train_jet = split_data(y_train, tx_train, ids_train, jet_num)
        columns_to_remove = analyse(tx_train_jet)
        tx_train_jet, header_train_jet = remove_columns(tx_train_jet, header_train, columns_to_remove)
        create_csv('train_jet_' + str(jet_num) + '.csv', y_train_jet, tx_train_jet, ids_train_jet, header_train_jet, False)
        print('\n... created train_jet_' + str(jet_num) + '.csv file.')
        print('\nProcessing test set with jet number = ' + str(jet_num) + '...')
        y_test_jet, tx_test_jet, ids_test_jet = split_data(y_test, tx_test, ids_test, jet_num)
        columns_to_remove = analyse(tx_test_jet)
        tx_test_jet, header_test_jet = remove_columns(tx_test_jet, header_test, columns_to_remove)
        create_csv('test_jet_' + str(jet_num) + '.csv', y_test_jet, tx_test_jet, ids_test_jet, header_test_jet, True)
        print('\n... created test_jet_' + str(jet_num) + '.csv file.')
        
def report_prediction_accuracy(y, tx, w_best, verbose = True):
    """
    Reports the percentage of correct predictions of a model that is applied
    on a set of labels.
    Args:
        y: numpy array of labels for testing purpose
        tx: numopy array of features in the learned data set
        w_best: the optimized weight vector of the model
    Returns:
        correct_percentage: the percentage of correct predictions of the model 
            when it is applied on the given test set of labels
    """
    predictions = tx.dot(w_best)
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = -1
    correct_percentage = np.sum(predictions == y) / float(len(predictions))
    if verbose:
        print('Percentage of correct predictions is: %', correct_percentage * 100)
    return correct_percentage

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

def cross_validation(y, augmented_tx, k_indices, k, lambda_, report_predictions = False):
    """
    Performs cross_validation for a specific test set from the partitioned set.
    Args:
        y: labels
        augmented_tx: augmented features
        k_indices: an array of k sub-indices that are randomly partitioned
        k: the test set that is kth partition 
        lambda_: regularization parameter for the ridge regression
    Returns:
        rmse_training: numeric value of the root mean squared error loss
            for the training set
        rmse_test: numeric value of the root mean squared error loss
            for the test set
    """
    y_test = y[k_indices[k]]
    y_training = np.delete(y, k_indices[k])
    augmented_tx_test = augmented_tx[k_indices[k]]
    augmented_tx_training = np.delete(augmented_tx, k_indices[k], axis = 0)
    w, loss_training = ridge_regression(y_training, augmented_tx_training, lambda_)
    if report_predictions:
        report_prediction_accuracy(y_test, augmented_tx_test, w, report_predictions)
    loss_test = compute_mse(compute_error_vector(y_test, augmented_tx_test, w))
    return compute_rmse(loss_training), compute_rmse(loss_test)

def report_prediction_accuracy_logistic(y, tx, w_best, verbose = True):
    """
    Reports the percentage of correct predictions of a model that is applied
    on a set of labels. This method specifically works for logistic regression
    since the prediction assumes that labels are between 0 and 1.
    Args:
        y: numpy array of labels for testing purpose
        tx: numpy array of features in the learned data set
        w_best: the optimized weight vector of the model
    Returns:
        correct_percentage: the percentage of correct predictions of the model 
            when it is applied on the given test set of labels
    """
    predictions = tx.dot(w_best)
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    correct_percentage = np.sum(predictions == y) / float(len(predictions))
    if verbose:
        print('Percentage of correct predictions is: %', correct_percentage * 100)
    return correct_percentage

def train_test_split(y, tx, ratio, seed=1):
    """
    Splits a given training data set to a test set and a training set,
    the sizes of the created sets are determined by the given ration.
    Args:
        y: numpy array of labels
        tx: numpy array of features 
        ratio: ratio of the created training set to the data set
        seed: the random seed
    Returns:
        y_training: numpy array of labels of the seperated training set 
        tx_training: numpy array of features of the seperated training set
        y_test: numpy array of labels of the seperated test set
        tx_test: numpy array of features of the seperated test set
    """
    np.random.seed(seed)
    permutation = np.random.permutation(len(y))
    shuffled_tx = tx[permutation]
    shuffled_y = y[permutation]
    split_position = int(len(y) * ratio)
    tx_training, tx_test = shuffled_tx[ : split_position], shuffled_tx[split_position : ]
    y_training, y_test = shuffled_y[ : split_position], shuffled_y[split_position : ]
    return y_training, tx_training, y_test, tx_test, 

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
    if mean_x is None:
        mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def min_max_normalization(x, min_of_x = None, max_of_x = None):
    """
    Normalizes the original data set using min max normalization.
    Args:
        x: data set to standardize
        min_of_x: minimum values of features, can be specified or computed
        max_of_x: maximum values of features, can be specified or computed
    Returns:
        x: min max normalized data set
        min_of_x: minimum values of features
        max_of_x: maximum values of features
    """
    if min_of_x is None:
        min_of_x = np.min(x, axis = 0)
    if max_of_x is None:
        max_of_x = np.max(x, axis = 0)
    return (x - (min_of_x)) / (max_of_x - min_of_x), min_of_x, max_of_x

def change_labels_logistic(y):
    """
    The labels in logistic regression are interpreted as probabilities,
    so this method transfers the labels to the range [0, 1]
    Args:
        y: given labels
    Returns:
        y_logistic: labels as probabilities
    """
    y[y == -1] = 0
    return y
    


    



