# Traffic Flow Prediction for Chicago Midterm Report
## Data Preprocessing

- Cleaned the dataset (e.g., replaced `nil`/missing values).
- Normalized numerical features to ensure consistent scale.
- Encoded categorical variables using **Label Encoding**.
- Parsed the `date_time` column into separate features:
  - `year`, `month`, `day`, and `time`
- Converted `time` into a float to represent the time of day numerically.
- Applied **Label Encoding** to `year`, `month`, and `day` for model compatibility.


## Linear Regression Model Summary
Using the pre-processed dataset, the linear regression model yielded a mean squared error of approximately 0.50 and an R-squared value of 0.51, the model explains about 51% of the variability in traffic volume, demonstrating a moderate fit that serves as a baseline for more complicated, non-linear models.

We also tried two regularization methods, ridge regression (with regularization parameter alpha=10.0) and lasso(with regularization parameter alpha=.001), which yielded similar results:

| Model                      | Mean Squared Error    | R-squared   |
|---------------------------|--------------------------------|-----------------|
| Linear Regression  | 0.50              | 0.51      |
| Ridge Regression   | 0.50              | 0.51      |
| Lasso                      | 0.50              | 0.51      |


## XGBoost Model Summary

### Data Preparation
- Loaded the preprocessed traffic data.
- Performed an 80/20 train-test split.

### Hyperparameter Tuning
- Conducted a grid search with 5-fold cross-validation on:
  - Regularization parameters: `alpha`, `lambda`, `gamma`
  - Learning rate: `eta`
  - Tree depth: `max_depth`
- **Best parameters found**:
  - `gamma`: 0  
  - `eta`: 0.3  
  - `max_depth`: 8  
  - `alpha`: 1  
  - `lambda`: 10

![Total Loss](figures/Total_Loss.png)
### Model Performance
- **Training RMSE**: 0.095  
- **Testing RMSE**: 0.158  
- RMSE plot shows that after ~5–6 boosting rounds, the model generalizes well.
- The small gap between training and testing RMSE indicates **no overfitting**.

### Feature Importance
- **By Weight**:
  - **Feature 10 (Temperature)** was the most frequently used for splits.
- **By Gain**:
  - **Feature 2 (Time of Day: Morning, Afternoon, Evening)** contributed the most to improving predictive accuracy.
- **Feature 3 (Holiday)** had minimal importance in both metrics, suggesting limited impact on traffic volume predictions.

### Conclusion
XGBoost demonstrated strong predictive power for traffic volume forecasting. It generalized well to unseen data and offered interpretability through feature importance analysis.
