# Two-stage TrAdaBoost.R2 algorithm from Pardoe's paper "Boosting for Regression Transfer (ICML 2010)"

## Description

* This is a boosting based transfer learning algorithm for regression tasks (TwoStageTrAdaBoostR2) that is proposed by Pardoe et al. in paper "Boosting for Regression Transfer (ICML 2010)". 
* The program is written in `scikit-learn` style and the structure is as follows: <br />
>Stage2_TrAdaBoostR2 <br />
>>| `__init__`<br />
>>|`fit` <br />
>>|`_stage2_adaboostR2`<br />
>>|`predict` <br />

>TwoStageTrAdaBoostR2
>>|`__init__`<br />
>>|`fit`<br />
>>|`_twostage_adaboostR2`<br />
>>|`beta_binary_search`<br />
>>|`predict`

* The first class `Stage2_TrAdaBoostR2` is a revised version of `AdaBoostRegressor` in `sklearn` package with the revision that the weights of certain data (source data) are never modified as discussed in Pardoe's paper. This class serves as the second-stage of the Two-Stage TrAdaBoost.R2 algorithm. 
* The second class `TwoStageTrAdaBoostR2` is the main class that implements the whole two stages of the transfer learnig algorithm. 
* Since the code is written in `sklearn` style, it is adaptable to any regressors in the `sklearn` packages, e.g., DecisionTreeRegressor.

## Usage
* `TwoStageTrAdaBoostR2`<br />
   Specify the settings of the algorithm including {base_estimator, sample_size, n_estimators, steps, fold, learning_rate, loss, random_state}. Some of these settings are the same with `AdaBoostRegressor` in `sklearn` package. The following settings are unique to the `TwoStageTrAdaBoostR2`:
   1. **sample_size** is a size two list of the sample size of the source data and target data, e.g., [100, 10]. 
   2. **steps** is the number of iteration steps *S* in Pardoe's paper. 
   3. **fold** controls the number of fold (*F*) for cross-validation in Pardoe's paper. 

*  `TwoStageTrAdaBoostR2.fit` <br />
   The inputs of the `fit` function include {X, y, sample_weight}
   1. **X** is the training input array including both source data and target data, X = [X_source, X_target] with shape = [sample_size[0]+sample_size[1], n_features]. 
   2. **y** is the training output  array including both source data and target data, y = [y_source, y_target] with shape = [sample_size[0]+sample_size[1]]. 
   3. **sample_weight** (optional) is the initial sample weight specified for the training data. If None, it will be set to the default equal weights. 
* `TwoStageTrAdaBoostR2.predict` <br />
Predict function for the prediction of unknown input area **X**
## Notes
An example is given by 'example1.py'
