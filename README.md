# CS506-Final-Project

## Project Goal:
Build a predictive model to estimate traffic volume based on historical data to
identify key factors influencing traffic volume, analyze trends and patterns in traffic data, and evaluate the performance of different machine learning models for this task.

##  Dataset Description
This dataset,  TrafficVolumeData.csv,  provides a comprehensive view of traffic flow of Chicago city, capturing a variety of factors that influence it. It consists of 33,744 entries, meticulously recorded to reflect the dynamics of traffic volume. Each entry is an amalgamation of weather conditions, temporal factors, and traffic metrics

##  Plan
Proposed Plan
The project will follow a structured data science workflow:

  ### Data Collection:
     - Load the dataset from traffic volume data csv on Kaggle.
     - Perform initial exploration to understand the structure and features.
  
  ### Data Preprocessing:
      - Handle missing values and outliers.
      - Encode categorical variables (e.g., weather conditions, holidays).
      - Feature engineering: Create new features such as:
      - Day of the week.
      - Time of day (e.g., morning, afternoon, evening).
      - Holiday flag.
      - Weather severity index.
  
  ### Exploratory Data Analysis (EDA):
     - Visualize trends in traffic volume over time.
     - Identify correlations between features and traffic volume.
     - Analyze the impact of weather, holidays, and time on traffic.
     
  ### Data Modelling (intended):
      We will train and evaluate multiple machine learning models, including:
      - Linear Regression (with ridge regression and lasso regression)
      - Decision Trees.
      - Random Forest.
      - Gradient Boosting (e.g., XGBoost, LightGBM).
      - Neural Networks (optional).
  
  ###  Data Visualization:
       Preliminary Visualization of the Data
        - Plot the data using T-SNE to visualize in a lower dimension
       Clustering
        -Plot a heatmap of most common features that appear in each cluster
      
## Test Plan:
We plan to split the data into training and testing sets (80/20). In terms of evaluation metrics, we intend to use Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared (R²).

To optimize the model performance, we intend to perform hyperparameter tuning using Grid Search or Random Search. We also plan to use cross-validation to ensure model robustness.

## Hyperlinks
https://www.kaggle.com/datasets/bobaaayoung/trafficvolumedatacsv
