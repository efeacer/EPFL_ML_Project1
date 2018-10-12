import proj1_helpers as p1
import implementations as ml
import numpy as np

y_train, x_train, ids = p1.load_csv_data('/Users/user/Desktop/ML_Project1Data/DataSet/train.csv')
x_train, mean_x, std_x = ml.standardize(x_train)

"""
w_best, loss = ml.least_squares(y_train, x_train)
print(loss)

y_test, tx_test, ids = p1.load_csv_data('/Users/user/Desktop/ML_Project1Data/DataSet/test.csv')
tx_test, _, _ = ml.standardize(tx_test, mean_x, std_x)
w_best, _ = ml.least_squares(y_train, x_train)
y_pred = p1.predict_labels(w_best, tx_test)
p1.create_csv_submission(ids, y_pred, 'submission8')
"""

"""
def get_best_lambda_using_cross_validation_demo(degree):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-11, -9, 40)
    k_indices = ml.build_k_indices(y_train, k_fold, seed)
    min_rmse_te = None
    lambda_best = None
    for lambda_ in lambdas:
        rmse_te = 0
        for k in range(k_fold):
            _, loss_te = ml.cross_validation(y_train, x_train, k_indices, k, lambda_, degree)
            rmse_te += loss_te 
        rmse_te /= k_fold
        if min_rmse_te == None or rmse_te < min_rmse_te:
            min_rmse_te = rmse_te
            lambda_best = lambda_
    print('Best lambda, best mse, with degree: ', lambda_best, (min_rmse_te ** 2) / 2, degree)
    return lambda_best, min_rmse_te

def get_best():
    degrees = [1]
    best_degree = None
    min_rmse = None
    best_lambda = None
    for degree in degrees:
        lambda_best, min_rmse_te = get_best_lambda_using_cross_validation_demo(degree)
        if  min_rmse == None or min_rmse_te < min_rmse:
            min_rmse = min_rmse_te
            best_degree = degree
            best_lambda = lambda_best
    print('Best lambda, best rmse, best degree: ', lambda_best, (min_rmse ** 2) / 2, best_degree)    
    return best_degree, best_lambda, min_rmse
            
best_degree, best_lambda, min_mse = get_best()
#x_train_augmented = ml.build_polynomial(x_train, best_degree)
#w_best = ml.ridge_regression(y_train, x_train_augmented, best_lambda)

#print(w_best)
"""

"""
l = loss
min_loss = np.sqrt(2 * loss)
print("Direct solution: ", min_w, l)
degrees = [3]
for _, degree in enumerate(degrees):
        tx_augmented = implementations.build_polynomial(tx_train, degree)
        lambdas = np.logspace(-15, 0, 100)
        min_w, loss = implementations.ridge_regression(y_train, tx_augmented, lambdas[0])
        min_loss = np.sqrt(2 * loss)
        lamdaaa = 0
        print(l)
        for _, lambda_ in enumerate(lambdas):
            # ridge regression
            w, loss = implementations.ridge_regression(y_train, tx_train, lambda_)
            curr_loss = np.sqrt(2 * loss)
            if (curr_loss < min_loss):
                min_loss = curr_loss
                l = loss
                lamdaaa = lambda_
                min_w = w
        print("Best lambda for current degree", min_w, l, lamdaaa)
print("Overall best", min_w, l)
"""

"""
lambdas = np.logspace(-20, 0, 100)
min_w, loss = ml.ridge_regression(y_train, x_train, lambdas[0])
min_loss = np.sqrt(2 * loss)
l = loss
lamdaaa = 0
print(l)
for ind, lambda_ in enumerate(lambdas):
        # ridge regression
        w, loss = ml.ridge_regression(y_train, x_train, lambda_)
        curr_loss = np.sqrt(2 * loss)
        if (curr_loss < min_loss):
            min_loss = curr_loss
            l = loss
            lamdaaa = lambda_
            min_w = w
print(min_w, l, lamdaaa)
"""
#0.3396868094770349
#0.33968680947703483 1.8329807108324375e-15
#0.3396868094770346 2.06913808111479e-14
#0.33968680948492724 
#0.33968680947703467 2.8480358684358048e-15
#0.33968680947703456 4.213321743847289e-14
#0.33968680947703445 2.2980599887588488e-14
#Best lambda, best mse, with degree:  3.3598182862837877e-10 0.3395657555468481 1
#Best lambda, best rmse, best degree:  4.923882631706731e-10 0.3395657514539506 1

"""
y_test, tx_test, ids = p1.load_csv_data('/Users/user/Desktop/ML_Project1Data/DataSet/test.csv')
tx_test, _, _ = ml.standardize(tx_test, mean_x, std_x)
tx_test_augmented = ml.build_polynomial(tx_test, 1)
x_train_augmented = ml.build_polynomial(x_train, 1)
w_best, _ = ml.ridge_regression(y_train, x_train_augmented, 4.923882631706731e-10)
y_pred = p1.predict_labels(w_best, tx_test_augmented)
p1.create_csv_submission(ids, y_pred, 'submission5')
"""





