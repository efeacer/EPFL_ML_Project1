import proj1_helpers
import implementations
import numpy as np

y_train, tx_train, ids = proj1_helpers.load_csv_data("/Users/user/Desktop/ML_Project1/DataSet/train.csv")
"""
w, loss = implementations.least_squares(y_train, tx_train)
print(w, loss)
w, loss = implementations.ridge_regression(y_train, tx_train, 0)
print(w, loss)
"""

lambdas = np.logspace(-15, -13, 1000)
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

#0.3396868094770349
#0.33968680947703483 1.8329807108324375e-15
#0.3396868094770346 2.06913808111479e-14
#0.33968680948492724 
#0.33968680947703467 2.8480358684358048e-15
#0.33968680947703456 4.213321743847289e-14
#0.33968680947703445 2.2980599887588488e-14

y_test, tx_test, ids = proj1_helpers.load_csv_data("/Users/user/Desktop/ML_Project1/DataSet/test.csv")
y_pred = proj1_helpers.predict_labels(min_w, tx_test)
proj1_helpers.create_csv_submission(ids, y_pred, "submission2")


