#  [markdown]
# # Running XGBoost on Preprocessed Data Set
# Load the data, perform a grid search, analyze results

# 
#Project Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 
#Load preprocessed the data
df = pd.read_csv("traffic_data_preprocessed.csv")
df.head()

# 
from sklearn.model_selection import train_test_split
#Creating a training and a testing split
X = df.drop(columns=["traffic_volume"]).values
y = df["traffic_volume"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# 
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# # preforming a grid search on various hyperparameters 
# parameters = { "booster"        : ["gbtree"],
#                "learning_rate"  : [0.3, 0.4, 0.5, 0.6],        #learning rate eta
#                "gamma"          : [0, 10, 20 , 30, 40],        #minimum loss reduction
#                "max_depth"      : [5, 6, 7, 8],                #depth of tree    
#                "reg_lambda"     : [0, 1, 5, 10, 50, 100],      #L2 regularization on weights
#                "reg_alpha"      : [0, 1, 5, 10, 50, 100],      #L1 regularizaton on weights
#              }

# #run the grid search to find the best parameters
# xgb = XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
# grid = GridSearchCV(estimator=xgb, param_grid=parameters, cv=5, verbose=1, n_jobs=-1)
# grid.fit(X_train, y_train)

# #
# #finding the parameters that generated the best XGBTree
# best_parameters = grid.best_params_
# print(best_parameters)


#from grid search best parameters were {'booster': 'gbtree', 'gamma': 0, 'learning_rate': 0.3, 'max_depth': 8, 'reg_alpha': 1, 'reg_lambda': 10}

xgb = XGBRegressor( booster = 'gbtree',
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    gamma = 0,
                    learning_rate = 0.3,
                    max_depth = 8, 
                    reg_alpha = 1, 
                    reg_lambda = 10,
                    random_state = 42
                    )

xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

#seeing how well our model trained
results = xgb.evals_result()
epochs = len(results['validation_0']['rmse'])
plt.figure(figsize=(6, 4))
plt.plot(range(epochs), results['validation_0']['rmse'], label = 'Train')
plt.plot(range(epochs), results['validation_1']['rmse'], label = 'Test')
plt.xlabel('Num of Boosting Rounds')
plt.ylabel('Root Mean Sqaured Error')
plt.title("RMSE vs Number of Rounds")
plt.legend()
plt.show()



#prediciting on training set and testing set
y_pred_training = np.array(xgb.predict(X_train))
y_pred_testing = np.array(xgb.predict(X_test))

#calculate the rmse of our model
y_train = np.array(y_train)
y_test = np.array(y_test) 

#function to calcuate rmse
def calculate_rmse(predictions : np.ndarray, ground_truth : np.ndarray) -> np.float64:
    n = predictions.shape[0]
    rmse = np.sqrt(np.sum(((predictions - ground_truth)**2)/n))
    return rmse

rmse_for_training_set = calculate_rmse(y_pred_training, y_train)
rmse_for_testing_set = calculate_rmse(y_pred_testing , y_test)

print("Training RMSE:", rmse_for_training_set)
print("Testing RMSE:", rmse_for_testing_set)

from xgboost import plot_importance

#importance of each feature (weight)
def plot_feature_importance(importance_type):
    plt.figure(figsize=(10, 8))
    plot_importance(xgb, importance_type=importance_type)
    plt.title(f'Importance of Features by {importance_type}')
    plt.show()

plot_feature_importance('weight')
plot_feature_importance('gain')


from sklearn.metrics import r2_score

#calculating R^2
y_pred = xgb.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

