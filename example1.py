"""
An example showing the usage of the TwoStageTrAdaBoostR2 algorithm. 
Example starts at line 396. 

"""

import numpy as np
import copy
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

##=============================================================================

#        copy the two classes from TwoStageTrAdaBoostR2 algorithm

##=============================================================================
class Stage2_TrAdaBoostR2:
    def __init__(self, 
                 base_estimator = DecisionTreeRegressor(max_depth=4), 
                 sample_size = None,  
                 n_estimators = 50, 
                 learning_rate = 1.,
                 loss = 'linear', 
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state


    def fit(self, X, y, sample_weight=None): 
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")
        
        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")
            
        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
    
        for iboost in range(self.n_estimators): # this for loop is sequential and does not support parallel(revison is needed if making parallel)
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._stage2_adaboostR2(
                    iboost,
                    X, y,
                    sample_weight)
            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        return self
    
    
    def _stage2_adaboostR2(self, iboost, X, y, sample_weight):
        
        estimator = copy.deepcopy(self.base_estimator) # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)
    
        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)
        
        self.estimators_.append(estimator)  # add the fitted estimator

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (sample_weight * error_vect).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)
        
        # avoid overflow of np.log(1. / beta)
        if beta < 1e-308:
            beta = 1e-308
        estimator_weight = self.learning_rate * np.log(1. / beta)
        
        # Boost weight using AdaBoost.R2 alg except the weight of the source data
        # the weight of the source data are remained
        source_weight_sum= np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
        target_weight_sum = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)
        
        if not iboost == self.n_estimators - 1:
            sample_weight[-self.sample_size[-1]:] *= np.power(
                    beta,
                    (1. - error_vect[-self.sample_size[-1]:]) * self.learning_rate)
            # make the sum weight of the source data not changing
            source_weight_sum_new = np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
            target_weight_sum_new = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)
            if source_weight_sum_new != 0. and target_weight_sum_new != 0.:
                sample_weight[:-self.sample_size[-1]] = sample_weight[:-self.sample_size[-1]]*source_weight_sum/source_weight_sum_new
                sample_weight[-self.sample_size[-1]:] = sample_weight[-self.sample_size[-1]:]*target_weight_sum/target_weight_sum_new

        return sample_weight, estimator_weight, estimator_error
    
    
    
    def predict(self, X):
        # Evaluate predictions of all estimators
        predictions = np.array([
                est.predict(X) for est in self.estimators_[:len(self.estimators_)]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]

        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]
    
    

class TwoStageTrAdaBoostR2:
    def __init__(self, 
                 base_estimator = DecisionTreeRegressor(max_depth=4), 
                 sample_size = None,  
                 n_estimators = 50, 
                 steps = 10, 
                 fold = 5, 
                 learning_rate = 1.,
                 loss = 'linear', 
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.steps = steps
        self.fold = fold
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state


    def fit(self, X, y, sample_weight=None): 
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")
        
        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")
            
        
        X_source = X[:-self.sample_size[-1]]
        y_source = y[:-self.sample_size[-1]]
        X_target = X[-self.sample_size[-1]:]
        y_target = y[-self.sample_size[-1]:]
        
        self.models_ = []
        self.errors_ = []
        for istep in range(self.steps):
            model = Stage2_TrAdaBoostR2(self.base_estimator,
                                        sample_size = self.sample_size, 
                                        n_estimators = self.n_estimators,
                                        learning_rate = self.learning_rate, loss = self.loss, 
                                        random_state = self.random_state)
            model.fit(X, y, sample_weight = sample_weight)
            self.models_.append(model)
            # cv training
            kf = KFold(n_splits = self.fold)
            error = []
            target_weight = sample_weight[-self.sample_size[-1]:]
            source_weight = sample_weight[:-self.sample_size[-1]]
            for train, test in kf.split(X_target):
                sample_size = [self.sample_size[0], len(train)]
                model = Stage2_TrAdaBoostR2(self.base_estimator,
                                        sample_size = sample_size, 
                                        n_estimators = self.n_estimators,
                                        learning_rate = self.learning_rate, loss = self.loss, 
                                        random_state = self.random_state)
                X_train = np.concatenate((X_source, X_target[train]))
                y_train = np.concatenate((y_source, y_source[train]))
                X_test = X_target[test]
                y_test = y_target[test]
                # make sure the sum weight of the target data do not change with CV's split sampling
                target_weight_train = target_weight[train]*np.sum(target_weight)/np.sum(target_weight[train])
                model.fit(X_train, y_train, sample_weight = np.concatenate((source_weight, target_weight_train)))
                y_predict = model.predict(X_test)
                error.append(mean_squared_error(y_predict, y_test))
            
            self.errors_.append(np.array(error).mean())

            sample_weight = self._twostage_adaboostR2(istep, X, y, sample_weight)
            
            if sample_weight is None:
                break
            if np.array(error).mean() == 0:
                break
            
            sample_weight_sum = np.sum(sample_weight)
            
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if istep < self.steps - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        return self
            
    
    def _twostage_adaboostR2(self, istep, X, y, sample_weight):
        
        estimator = copy.deepcopy(self.base_estimator) # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)
    
        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)
        

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Update the weight vector
        beta = self._beta_binary_search(istep, sample_weight, error_vect, stp = 1e-30)

        if not istep == self.steps - 1:
            sample_weight[:-self.sample_size[-1]] *= np.power(
                    beta,
                    (error_vect[:-self.sample_size[-1]]) * self.learning_rate)
        return sample_weight
    
    
    def _beta_binary_search(self, istep, sample_weight, error_vect, stp):
        # calculate the specified sum of weight for the target data
        n_target = self.sample_size[-1]
        n_source = np.array(self.sample_size).sum() - n_target
        theoretical_sum = n_target/(n_source+n_target) + istep/(self.steps-1)*(1-n_target/(n_source+n_target))
        # for the last iteration step, beta is 0.
        if istep == self.steps - 1:
            beta = 0.
            return beta
        # binary search for beta
        L = 0.
        R = 1.
        beta = (L+R)/2
        sample_weight_ = copy.deepcopy(sample_weight)
        sample_weight_[:-n_target] *= np.power(
                    beta,
                    (error_vect[:-n_target]) * self.learning_rate)
        sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
        updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
        
        while np.abs(updated_weight_sum - theoretical_sum) > 0.01:
            if updated_weight_sum < theoretical_sum:
                R = beta - stp
                if R > L:
                    beta = (L+R)/2
                    sample_weight_ = copy.deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                                beta,
                                (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    print("At step:", istep+1)
                    print("Binary search's goal not meeted! Value is set to be the available best!")
                    print("Try reducing the search interval. Current stp interval:", stp)
                    break
                
            elif updated_weight_sum > theoretical_sum:
                L = beta + stp
                if L < R:
                    beta = (L+R)/2
                    sample_weight_ = copy.deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                                beta,
                                (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    print("At step:", istep+1)
                    print("Binary search's goal not meeted! Value is set to be the available best!")
                    print("Try reducing the search interval. Current stp interval:", stp)
                    break
        return beta
        
        
    
    def predict(self, X):
        # select the model with the least CV error
        fmodel = self.models_[np.array(self.errors_).argmin()]
        predictions = fmodel.predict(X)
        return predictions 
    
##=============================================================================
        
#                            end copying the two classes

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


