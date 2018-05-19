"""
An example showing the usage of the TwoStageTrAdaBoostR2 algorithm. 

"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 # import the two-stage algorithm
##=============================================================================

#                                Example 1

##=============================================================================
        
# 1. define the data generating function
def response(x, d, random_state):
    """
    x is the input variable
    d controls the simularity of different tasks
    """
    a1 = np.random.normal(1, 0.1*d)
    a2 = np.random.normal(1, 0.1*d)
    b1 = np.random.normal(1, 0.1*d)
    b2 = np.random.normal(1, 0.1*d)
    c1 = np.random.normal(1, 0.05*d)
    c2 = np.random.normal(1, 0.05*d)
    y = a1*np.sin(b1*x + c1).ravel() + a2*np.sin(b2*6 * x + c2).ravel() + random_state.normal(0, 0.1, x.shape[0])
    return y

#==============================================================================
    
#     2. decide the degree of similarity of multiple data sources using d
    
d = 0.5
#==============================================================================
rng = np.random.RandomState(1)

# 3.1 create source data and target data
n_source1 = 100
x_source1 = np.linspace(0, 6, n_source1)[:, np.newaxis]
y_source1 = response(x_source1, d, rng)
n_source2 = 100
x_source2 = np.linspace(0, 6, n_source2)[:, np.newaxis]
y_source2 = response(x_source2, d, rng)
n_source3 = 100
x_source3 = np.linspace(0, 6, n_source3)[:, np.newaxis]
y_source3 = response(x_source3, d, rng)
n_source4 = 100
x_source4 = np.linspace(0, 6, n_source4)[:, np.newaxis]
y_source4 = response(x_source4, d, rng)
n_source5 = 100
x_source5 = np.linspace(0, 6, n_source5)[:, np.newaxis]
y_source5 = response(x_source5, d, rng)

# 3.2 create target data (n_target_train and n_target_test are the sample size of train and test datasets)
a1 = np.random.normal(1, 0.1*d)
a2 = np.random.normal(1, 0.1*d)
b1 = np.random.normal(1, 0.1*d)
b2 = np.random.normal(1, 0.1*d)
c1 = np.random.normal(1, 0.05*d)
c2 = np.random.normal(1, 0.05*d)

# target_train
#==============================================================================

n_target_train = 15

#==============================================================================
x_target_train = np.linspace(0, 6, n_target_train)[:, np.newaxis]
y_target_train = a1*np.sin(b1*x_target_train + c1).ravel() + a2*np.sin(b2*6 * x_target_train + c2).ravel() + rng.normal(0, 0.1, x_target_train.shape[0])

# target_test
n_target_test = 600
x_target_test = np.linspace(0, 6, n_target_test)[:, np.newaxis]
y_target_test = a1*np.sin(b1*x_target_test + c1).ravel() + a2*np.sin(b2*6 * x_target_test + c2).ravel() + rng.normal(0, 0.1, x_target_test.shape[0])

# 3.3 plot the generated data
plt.figure()
plt.plot(x_source1, y_source1, c="r", label="source1", linewidth=1)
plt.plot(x_source2, y_source2, c="y", label="source2", linewidth=1)
plt.plot(x_source3, y_source3, c="g", label="source3", linewidth=1)
plt.plot(x_source4, y_source4, c="c", label="source4", linewidth=1)
plt.plot(x_source5, y_source5, c="m", label="source5", linewidth=1)
plt.plot(x_target_test, y_target_test, c="b", label="target_test", linewidth=0.5)
plt.scatter(x_target_train, y_target_train, c="k", label="target_train")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Multiple datasets")
plt.legend()
plt.show()

# 4. transfer learning regressiong for the target_train data
# 4.1 data combination and initial setting specification
X = np.concatenate((x_source1, x_source2, x_source3, x_source4, x_source5, x_target_train))
y = np.concatenate((y_source1, y_source2, y_source3, y_source4, y_source5, y_target_train))
sample_size = [n_source1+n_source2+n_source3+n_source4+n_source5, n_target_train]

#==============================================================================

n_estimators = 100
steps = 10
fold = 5
random_state = np.random.RandomState(1)

#==============================================================================

# 4.2 TwoStageAdaBoostR2
regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                      n_estimators = n_estimators, sample_size = sample_size, 
                      steps = steps, fold = fold, 
                      random_state = random_state)
regr_1.fit(X, y)
y_pred1 = regr_1.predict(x_target_test)

# 4.3 As comparision, use AdaBoostR2 without transfer learning
#==============================================================================
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),
                          n_estimators = n_estimators)
#==============================================================================
regr_2.fit(x_target_train, y_target_train)
y_pred2 = regr_2.predict(x_target_test)

# 4.4 Plot the results
plt.figure()
plt.scatter(x_target_train, y_target_train, c="k", label="target_train")
plt.plot(x_target_test, y_target_test, c="b", label="target_test", linewidth=0.5)
plt.plot(x_target_test, y_pred1, c="r", label="TwoStageTrAdaBoostR2", linewidth=2)
plt.plot(x_target_test, y_pred2, c="y", label="AdaBoostR2", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Two-stage Transfer Learning Boosted Decision Tree Regression")
plt.legend()
plt.show()
# 4.5 Calculate mse
mse_twostageboost = mean_squared_error(y_target_test, y_pred1)   
mse_adaboost = mean_squared_error(y_target_test, y_pred2)
print("MSE of regular AdaboostR2:", mse_adaboost)
print("MSE of TwoStageTrAdaboostR2:", mse_twostageboost)
#==============================================================================


