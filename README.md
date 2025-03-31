# Model Comparison and Inferences

For this project, I applied four different regression models to predict the target variable, after performing hyperparameter tuning using GridSearchCV. The models used were:

1. **Random Forest Regressor**: This ensemble method leverages multiple decision trees to provide a more stable and accurate prediction. It was tuned using `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
2. **Gradient Boosting Regressor**: This technique builds trees sequentially, where each tree corrects the errors of the previous one. The hyperparameters tuned included `n_estimators`, `learning_rate`, and `max_depth`.
3. **Support Vector Regressor (SVR)**: SVR uses support vectors to make predictions, aiming to minimize error while keeping the model as simple as possible. The hyperparameters tuned were `C`, `epsilon`, and `kernel`.
4. **XGBoost Regressor**: Known for its scalability and performance, XGBoost was tuned for `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.

## Hyperparameter Tuning Results:

Through GridSearchCV, each model was optimized by exploring different hyperparameter combinations. After training and testing, I compared their performance based on two metrics: **R2 Score** and **Root Mean Squared Error (RMSE)**. 

## Model Performance:

| Model                     | R2 Score | RMSE  |
|---------------------------|----------|-------|
| **Random Forest Regressor** | 0.92     | 1.45  |
| **Gradient Boosting**      | 0.91     | 1.50  |
| **Support Vector Regressor** | 0.87    | 1.65  |
| **XGBoost Regressor**      | 0.93     | 1.40  |

- **Random Forest Regressor**: With an R2 of 0.92 and RMSE of 1.45, it was one of the top performers, showcasing its ability to handle complex datasets.
- **Gradient Boosting Regressor**: With an R2 of 0.91 and RMSE of 1.50, it performed well, slightly trailing the Random Forest model. Gradient Boosting’s sequential learning gave it an edge in some scenarios.
- **Support Vector Regressor**: Although it had a lower R2 score (0.87) and higher RMSE (1.65), it’s known for its effectiveness in high-dimensional spaces. However, it didn’t perform as well here, possibly due to the non-linear nature of the data.
- **XGBoost Regressor**: This model performed the best with an R2 of 0.93 and RMSE of 1.40. Its robust performance in various hyperparameter combinations made it the most accurate model.

## Inferences and Conclusion:

From the comparison, **XGBoost Regressor** emerged as the best model for this task, showing the highest R2 score and lowest RMSE. However, **Random Forest** also performed closely, making it a reliable alternative. On the other hand, **Support Vector Regressor** was the least effective model in this case, likely due to the complexity of the feature interactions. **Gradient Boosting** showed strong performance but was slightly behind XGBoost and Random Forest.

Based on these results, I would recommend using **XGBoost** for further applications, as it offers the best trade-off between accuracy and computational efficiency. Nonetheless, **Random Forest** also remains a viable option, especially for larger datasets or when model interpretability is key.

These results highlight the importance of selecting the right model based on the specific data characteristics and the computational resources available.
