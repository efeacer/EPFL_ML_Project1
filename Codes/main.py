import proj1_helpers
import implementations
import numpy as np

y_train, tx_train, ids = proj1_helpers.load_csv_data("/Users/user/Desktop/ML_Project1Data/DataSet/train.csv")
tx_train, mean_x, std_x = implementations.standardize(tx_train)

min_w, loss = implementations.ridge_regression(y_train, tx_train, 0)
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
min_w, loss = implementations.ridge_regression(y_train, tx_train, lambdas[0])
min_loss = np.sqrt(2 * loss)
l = loss
lamdaaa = 0
print(l)
for ind, lambda_ in enumerate(lambdas):
        # ridge regression
        w, loss = implementations.ridge_regression(y_train, tx_train, lambda_)
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

y_test, tx_test, ids = proj1_helpers.load_csv_data("/Users/user/Desktop/ML_Project1Data/DataSet/test.csv")
tx_test, _, _ = implementations.standardize(tx_test, mean_x, std_x)
y_pred = proj1_helpers.predict_labels(min_w, tx_test)
proj1_helpers.create_csv_submission(ids, y_pred, "submission3")



