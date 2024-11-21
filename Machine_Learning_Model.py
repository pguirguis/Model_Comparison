# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:09:55 2023

@author: peter
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split




data = pd.read_excel(r'Molec.xlsx', "Fitting", engine= 'openpyxl', header= 1)

data_pred = pd.read_excel(r'Molec.xlsx', "Predict", engine= 'openpyxl', header= 1)

x = np.vstack((data["Other Carbs wt%"].to_numpy(),data["Starch wt%"].to_numpy(),data["Cellulose wt%"].to_numpy(),data["Protein wt%"].to_numpy(),data["Hemicellulose wt%"].to_numpy(),data["Lipids wt%"].to_numpy(),data["Lignin wt%"].to_numpy(),data["Sa wt%"].to_numpy(),data["AA wt%"].to_numpy(),data["FA wt%"].to_numpy(),data["Ph wt%"].to_numpy(),data["Ash wt%"].to_numpy(),data["Solid content (w/w) %"].to_numpy(),data["Total time (min)"].to_numpy(),data["Temperature (C)"].to_numpy()))

X = np.transpose(x)

y = data["Biocrude wt%"].to_numpy()

x_pred = np.vstack((data_pred["Other Carbs wt%"].to_numpy(),data_pred["Starch wt%"].to_numpy(),data_pred["Cellulose wt%"].to_numpy(),data_pred["Protein wt%"].to_numpy(),data_pred["Hemicellulose wt%"].to_numpy(),data_pred["Lipids wt%"].to_numpy(),data_pred["Lignin wt%"].to_numpy(),data_pred["Sa wt%"].to_numpy(),data_pred["AA wt%"].to_numpy(),data_pred["FA wt%"].to_numpy(),data_pred["Ph wt%"].to_numpy(),data_pred["Ash wt%"].to_numpy(),data_pred["Solid content (w/w) %"].to_numpy(),data_pred["Total time (min)"].to_numpy(),data_pred["Temperature (C)"].to_numpy()))

X_pred = np.transpose(x_pred)

y_pred = data_pred["Biocrude wt%"].to_numpy()

out = pd.ExcelWriter('Results2.xlsx')

def analysis(exp, pred, N_ks,mod_name,suffix,count):
    Res = np.subtract(exp, pred)
    
    rel_abs_Res = np.abs(np.divide(Res,exp))
    RSS = sum([item*item for item in Res if str(item) != "nan"  and str(item) != "inf" and type(item) != bool ])
    res_clean = [item for item in Res if str(item) != "nan"  and str(item) != "inf"  and type(item) != bool]
    rel_clean = [item for item in rel_abs_Res if str(item) != "nan"  and str(item) != "inf"  and type(item) != bool]
    abs_res = [abs(item) for item in res_clean ]
    n_point = len(res_clean)
    # print(n_point)
    med_Res = np.nanmedian(res_clean)
    # mean_Res = np.nanmean(res_clean)
    std_Res = np.std(res_clean)
    mean_abs_Res = np.nanmean(abs_res)
    mape = np.nanmean(rel_clean)*100
    med_abs_res = np.nanmedian(abs_res)
    med_rel = np.nanmedian(rel_clean)*100
    AIC = 2*N_ks + np.log(RSS/n_point)*n_point
    BIC = N_ks*np.log(n_point) +np.log(RSS/n_point)*n_point
    num_5 = len([item for item in abs_res  if item <= 5])/n_point*100
    num_10 = len([item for item in abs_res if item <= 10])/n_point*100
    num_75 = len([item for item in abs_res if item >= 75])/n_point*100
    num_100_rel = len([item for item in rel_clean if item >= 1])/n_point*100
    med_Res, std_Res, mean_abs_Res, mape, med_abs_res, med_rel, BIC, AIC, n_point, num_5, num_10, num_75, num_100_rel = np.round(med_Res, decimals=5), np.round(std_Res, decimals=5), np.round(mean_abs_Res, decimals=5),  np.round(mape, decimals=5), np.round(med_abs_res, decimals=5),np.round(med_rel, decimals=5), np.round(BIC, decimals=5) , np.round(AIC, decimals=5) , n_point, np.round(num_5, decimals=5), np.round(num_10, decimals=5), np.round(num_75, decimals=5), np.round(num_100_rel, decimals=5)
    df = pd.DataFrame([mod_name+"_"+suffix,n_point,N_ks,med_Res, mean_abs_Res,mape,AIC,  num_5, num_10])
    df = df.T
    df.to_excel(out,startrow=count, index=False, header=False)
    print([n_point,N_ks, med_Res, mean_abs_Res, med_abs_res,  mape, AIC, num_5, num_10])
    return [n_point,N_ks, med_Res, mean_abs_Res, med_abs_res,  mape, AIC, num_5, num_10]

df = pd.DataFrame(["Model","N point","N Parameters","Mean Res", "Mean Absolute Res", "MAPE",  "AIC",  "% < 5 wt%", "% < 10 wt%"])
df = df.T
df.to_excel(out,startrow=0, index=False, header=False)


""" Ridge regression """

print("\n")
print("Ridge regression")

from sklearn.linear_model import Ridge

count = 1
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X2, X_test2, y2, y_test2 = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the Ridge Regression model
# Regularization strength (adjust as needed)
ridge_model = Ridge(alpha=1)

# Fit the model to the training data
ridge_model.fit(X_train, y_train)

# Finding training set predictions
y_pred_train = ridge_model.predict(X_train)

# Make predictions on the test set
y_pred_test = ridge_model.predict(X_test)

# Making final predictions for prediciton set
y_pred_final = ridge_model.predict(X_pred)


# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)

# print(f"Mean Squared Error: {mse}\n")

num_cof = np.sum(ridge_model.coef_ != 0) + 1

analysis(y_train, y_pred_train,num_cof,"Ridge_regression","Training_1",count)
analysis(y_test, y_pred_test,num_cof,"Ridge_regression","Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,"Ridge_regression","Pred_1",count + 30)

ridge_model.fit(X, y)

# Finding training set predictions
y_pred_train = ridge_model.predict(X)

# Making final predictions for prediciton set
y_pred_final = ridge_model.predict(X_pred)

num_cof = np.sum(ridge_model.coef_ != 0) + 1

analysis(y, y_pred_train,num_cof,"Ridge_regression","Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,"Ridge_regression","Pred_2",count + 60)

""" Lasso Regression """

print("\n")
print("Lasso Regression")

count += 1

from sklearn.linear_model import Lasso

# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Lasso Regression model

lasso_model = Lasso(alpha=1)

# Fit the model to the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_train = lasso_model.predict(X_train)

y_pred_test = lasso_model.predict(X_test)

y_pred_final = lasso_model.predict(X_pred)

# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)


# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,lasso_model.coef_.shape[0]))

model_name = "Lasso_regression"

num_cof = np.sum(lasso_model.coef_ != 0) + 1

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)

lasso_model.fit(X, y)

y_pred_train = lasso_model.predict(X)
y_pred_final = lasso_model.predict(X_pred)

num_cof = np.sum(lasso_model.coef_ != 0) + 1

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)

""" Elastic Net Regression """

print("\n")
print("Elastic Net Regression")

count += 1

from sklearn.linear_model import ElasticNet

# Split the data into training and testing sets
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Elastic Net Regression model
alpha = 1  # Regularization strength (adjust as needed)
l1_ratio = 1  # L1 ratio (0: Ridge, 1: Lasso, in between: Elastic Net)
elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

# Fit the model to the training data
elastic_net_model.fit(X_train, y_train.ravel())

# Make predictions on the test set
y_pred_train = elastic_net_model.predict(X_train)

y_pred_test = elastic_net_model.predict(X_test)

y_pred_final = elastic_net_model.predict(X_pred)

# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,elastic_net_model.coef_.shape[0]))

model_name = "Elastic_Net_Regression"

num_cof = np.sum(elastic_net_model.coef_ != 0) + 1

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)

elastic_net_model.fit(X, y.ravel())

y_pred_train = elastic_net_model.predict(X)
y_pred_final = elastic_net_model.predict(X_pred)

num_cof = np.sum(elastic_net_model.coef_ != 0) + 1

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)


""" Random Forest Algorthem """

print("\n")
print("Random Forest Algorthem")

count += 1

from sklearn.ensemble import  RandomForestRegressor


# Sample data (replace this with your actual dataset)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Initialize the RandomForestRegressor
reg = RandomForestRegressor(n_estimators=12, random_state=23,criterion='squared_error')

# Fit the model to the training data
reg.fit(X_train, y_train)


# Make predictions on the testing set
y_pred_train = reg.predict(X_train)

y_pred_test = reg.predict(X_test)

y_pred_final = reg.predict(X_pred)



model_name = "Random_Forest_Algorthem"

num_cof = sum(tree.get_n_leaves() for tree in reg.estimators_)

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)


reg.fit(X, y)


y_pred_final = reg.predict(X_pred)

num_cof = sum(tree.get_n_leaves() for tree in reg.estimators_)

y_pred_train = reg.predict(X)

y_pred_final = reg.predict(X_pred)

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)
        


# analysis(y_pred, y_pred_final,num_cof,model_name,"Testing_2",count + 15)

# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,len(reg.feature_importances_)))

""" Gradient Boosting Regression """

count += 1

print("\n")
print("Gradient Boosting Regression")

from sklearn.ensemble import GradientBoostingRegressor

# Split the data into training and testing sets
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regression model
learning_rate = 1.17  # Adjust as needed
n_estimators = 22   # Number of boosting stages to be run
gbr_model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)

# Fit the model to the training data
gbr_model.fit(X_train, y_train.ravel())  # ravel() to convert y to a 1D array

# Make predictions on the test set
y_pred_train = gbr_model.predict(X_train)

y_pred_test = gbr_model.predict(X_test)

y_pred_final = gbr_model.predict(X_pred)


model_name = "Gradient_Boosting_Regression"

num_cof = sum(estimator.get_n_leaves() for estimator in gbr_model.estimators_[:, 0])

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)

gbr_model.fit(X, y.ravel())

y_pred_train = gbr_model.predict(X)
y_pred_final = gbr_model.predict(X_pred)

num_cof = sum(estimator.get_n_leaves() for estimator in gbr_model.estimators_[:, 0])

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)



""" Support Vector Machine"""

print("Support Vector Machine\n")

count += 1

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


# Example for regression task
# def svm_regression_example(X, y):
# Split the data into training and testing sets
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Regressor (SVR)
reg = SVR(kernel='poly', C=3.5)

# Fit the model to the training data
reg.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_train = reg.predict(X_train)

y_pred_test = reg.predict(X_test)

y_pred_final = reg.predict(X_pred)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,reg.support_vectors_.shape[0] * (reg.degree + 1)))

model_name = "Support_Vector_Machine"

num_cof = len(reg.support_) + 1

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)

reg.fit(X, y)
y_pred_train = reg.predict(X)
y_pred_final = reg.predict(X_pred)

num_cof = len(reg.support_) + 1

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)

""" k-Nearest Neighbors (k-NN)"""

print("\n")
print("k-Nearest Neighbors (k-NN)")

count += 1

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Initialize the KNeighborsRegressor
reg = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors (k) here

# Fit the model to the training data
reg.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_train = reg.predict(X_train)

y_pred_test = reg.predict(X_test)

y_pred_final = reg.predict(X_pred)

# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,len(reg._fit_X)))

model_name = "k-Nearest_Neighbors_(k-NN)"

num_cof = reg.n_neighbors

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)


# Fit the model to the training data
reg.fit(X, y)

# Make predictions on the testing set
y_pred_train = reg.predict(X)
y_pred_final = reg.predict(X_pred)

num_cof = reg.n_neighbors

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)


""" Bayesian Regression """

print("\n")
print("Bayesian Regression")

count += 1

from sklearn.linear_model import BayesianRidge


# Split the data into training and testing sets
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Bayesian Ridge Regression model
bayesian_ridge_model = BayesianRidge()

# Fit the model to the training data
bayesian_ridge_model.fit(X_train, y_train.ravel())  # ravel() is used to flatten y

# Make predictions on the test set
y_pred_train, y_std = bayesian_ridge_model.predict(X_train, return_std=True)

y_pred_test, y_std = bayesian_ridge_model.predict(X_test, return_std=True)

y_pred_final, y_std = bayesian_ridge_model.predict(X_pred, return_std=True)

# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,bayesian_ridge_model.coef_.shape[0]))


model_name = "Bayesian_Regression"

num_cof = np.sum(bayesian_ridge_model.coef_ != 0) + 1 + 2

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)

# Fit the model to the training data
bayesian_ridge_model.fit(X, y.ravel())  # ravel() is used to flatten y

# Make predictions on the test set
y_pred_train, y_std = bayesian_ridge_model.predict(X, return_std=True)
y_pred_final, y_std = bayesian_ridge_model.predict(X_pred, return_std=True)

num_cof = np.sum(bayesian_ridge_model.coef_ != 0) + 1 + 2

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)

""" Gaussian Process Regression """

print("\n")
print("Gaussian Process Regression")

count += 1

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV

# Split the data into training and testing sets
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Gaussian Process kernel (RBF kernel with a constant term)
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

# Initialize the Gaussian Process Regressor
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=42)

param_grid = {"kernel__k1__constant_value": [0.1, 1, 10],
              "kernel__k2__length_scale": [0.1, 1, 10]}

grid_search = GridSearchCV(gpr, param_grid, cv=5, scoring='neg_mean_absolute_error')

# Fit the model to the training data
grid_search.fit(X_train, y_train)

gpr = grid_search.best_estimator_

# Make predictions on the test set
y_pred_train = gpr.predict(X_train, return_std=False)

y_pred_test = gpr.predict(X_test, return_std=False)

y_pred_final = gpr.predict(X_pred, return_std=False)

# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,len(gpr.kernel.theta)))

model_name = "Gaussian_Process_Regression"

num_cof = len(gpr.kernel_.get_params()) +1

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)

gpr.fit(X, y)

# Make predictions on the test set
y_pred_train = gpr.predict(X, return_std=False)

y_pred_final = gpr.predict(X_pred, return_std=False)

num_cof = len(gpr.kernel_.get_params()) + 1

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)


""" Huber Regression """

print("\n")
print("Huber Regression")

count += 1

from sklearn.linear_model import HuberRegressor

# Split the data into training and testing sets
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Huber Regression model
epsilon = 1.358  # Tuning parameter controlling the number of outliers
huber_model = HuberRegressor(epsilon=epsilon)

# Fit the model to the training data
huber_model.fit(X_train, y_train.ravel())

# Make predictions on the test set
y_pred_train = huber_model.predict(X_train)

y_pred_test = huber_model.predict(X_test)

y_pred_final = huber_model.predict(X_pred)

# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,huber_model.coef_.shape[0]))

model_name = "Huber_Regression"

num_cof = X_train.shape[1] + 1

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)

huber_model.fit(X, y.ravel())

# Make predictions on the test set
y_pred_train = huber_model.predict(X)

y_pred_final = huber_model.predict(X_pred)

num_cof = X_train.shape[1] + 1

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)

""" Decision Forest Regression"""

print("\n")
print("Decision Forest Regression")

count += 1

from sklearn.tree import DecisionTreeRegressor
from collections import Counter

# Initialize the RandomForestRegressor
reg = DecisionTreeRegressor(criterion='absolute_error',random_state=3,max_depth=24)

# Fit the model to the training data
reg.fit(X_train, y_train)

y_pred_train = reg.predict(X_train)

y_pred_test = reg.predict(X_test)

y_pred_final = reg.predict(X_pred)



model_name = "Decision_Tree_Algorthem"

num_cof = reg.get_n_leaves()

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)

reg.fit(X, y)

num_cof = reg.get_n_leaves()

y_pred_train = reg.predict(X)

main_calc = y_pred_train

res1 = np.subtract(y,y_pred_train)

data_pred = pd.read_excel(r'Molec.xlsx', "Predict", engine= 'openpyxl', header= 1)

data_pred = data_pred[data_pred["Total Lignin wt%"] == 0]
x_pred = np.vstack((data_pred["Other Carbs wt%"].to_numpy(),data_pred["Starch wt%"].to_numpy(),data_pred["Cellulose wt%"].to_numpy(),data_pred["Protein wt%"].to_numpy(),data_pred["Hemicellulose wt%"].to_numpy(),data_pred["Lipids wt%"].to_numpy(),data_pred["Lignin wt%"].to_numpy(),data_pred["Sa wt%"].to_numpy(),data_pred["AA wt%"].to_numpy(),data_pred["FA wt%"].to_numpy(),data_pred["Ph wt%"].to_numpy(),data_pred["Ash wt%"].to_numpy(),data_pred["Solid content (w/w) %"].to_numpy(),data_pred["Total time (min)"].to_numpy(),data_pred["Temperature (C)"].to_numpy()))

X_pred = np.transpose(x_pred)

y_pred = data_pred["Biocrude wt%"].to_numpy()

y_pred_final = reg.predict(X_pred)

main_calc_pred = y_pred_final


res = np.subtract(y_pred,y_pred_final)
leaf_indices = reg.apply(X)
leaf_counts = Counter(leaf_indices)

# Identify and count leaves with exactly one data point
leaves_with_one_data_point = sum(1 for count in leaf_counts.values() if count == 1)
print(analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45))
print(analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60))
        


""" RANSAC Regression """

print("\n")
print("RANSAC Regression")

count += 1

from sklearn.linear_model import RANSACRegressor

# Split the data into training and testing sets
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RANSAC Regression model
ransac_model = RANSACRegressor()

# Fit the model to the training data
ransac_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_train = ransac_model.predict(X_train)

y_pred_test = ransac_model.predict(X_test)

y_pred_final = ransac_model.predict(X_pred)

# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,ransac_model.estimator_.coef_.shape[0]+1))

model_name = "RANSAC_Regression"

num_cof = X_train.shape[1] + 1

analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count)
analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30)

# Fit the model to the training data
ransac_model.fit(X, y)

# Make predictions on the test set
y_pred_train = ransac_model.predict(X)

y_pred_final = ransac_model.predict(X_pred)

num_cof = X_train.shape[1] + 1

analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45)
analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60)

""" Generalized Additive Model """

print("\n")
print("Generalized Additive Model")

count += 1

from pygam import LinearGAM

# Split the data into training and testing sets
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Generalized Additive Model
gam_model = LinearGAM()

# Fit the model to the training data
gam_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_train = gam_model.predict(X_train)

y_pred_test = gam_model.predict(X_test)

y_pred_final = gam_model.predict(X_pred)

# Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}\n")
# print(analysis(y_test, y_pred,gam_model.coef_.shape[0]))

model_name = "Generalized_Additive_Model"

num_cof = len(gam_model.coef_)

print(analysis(y_train, y_pred_train,num_cof,model_name,"Training_1",count))
print(analysis(y_test, y_pred_test,num_cof,model_name,"Testing_1",count +15))
print(analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_1",count + 30))

gam_model.fit(X, y)

# Make predictions on the test set
y_pred_train = gam_model.predict(X)

y_pred_final = gam_model.predict(X_pred)

num_cof = len(gam_model.coef_)

print(analysis(y, y_pred_train,num_cof,model_name,"Training_2",count + 45))
print(analysis(y_pred, y_pred_final,num_cof,model_name,"Pred_2",count + 60))



# # out.save()
out.close()