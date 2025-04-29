# Final Report

---

# Model Progression and Inferences

Following the feedback from the midterm report, I further developed the same three models – MLP Regressor, Logistic Regression, and Decision Tree Classifier – by increasing their complexity. The goal was to observe whether these models could improve their performance or demonstrate signs of overfitting.

---

## MLP Regressor

The MLP Regressor was deepened by adding more hidden layers (512, 256, 128), increasing the number of iterations to 1000, and disabling early stopping. The goal was to allow the model greater capacity to fit the training data.

- **R² Score:** 0.720 (previous value: 0.493) 
    The model explained approximately 72% of the variance in traffic volume, a substantial improvement from the earlier version.

- **Mean Squared Error (MSE):**  0.685 (previous value: 1.243)  
    The MSE nearly halved from the previous version, indicating more accurate predictions.

This suggests that increasing network complexity helped the model capture more underlying patterns in the data. However, the improvement may come at the cost of overfitting, as generalization to unseen data would need to be further validated.

---

## Logistic Regression

To improve performance, the Logistic Regression model was updated by scaling all features using StandardScaler and increasing the regularization parameter C to 10. This linear classification model was evaluated with the following performance metrics:

- **Accuracy:** 39.8%  (previous value: 39.8%)
    Despite feature scaling and weaker regularization, the model’s overall accuracy remained essentially unchanged compared to the midterm model.

- **Confusion Matrix and Classification Report:**  
    The progressed model continued to misclassify a significant number of instances, and class imbalance persisted. Precision and recall remained low to moderate across classes: precision values ranged from 0.35 to 0.47. and recall values ranged from 0.21 to 0.54. F1-scores remained moderate, between 0.28 and 0.50, across most classes.
  
These results reinforce the limitations of logistic regression for this task: the model’s linear decision boundaries are insufficient to capture the underlying complexity and non-linear relationships present in the traffic dataset, even after addressing feature scaling and regularization.

---

## Decision Tree Classifier

The Decision Tree model was allowed to grow fully by removing depth constraints and minimizing the stopping criteria.

- **Accuracy:** 81.5% (previous value: 63.7%)
    The model showed a significant boost in accuracy compared to the midterm version.

- **Confusion Matrix and Classification Report:** 
    Precision and recall values ranged from 0.79 to 0.85, with F1-scores around 0.80 for all classes, indicating balanced class-wise performance.

The improved accuracy suggests that the fully grown tree captured complex patterns in the data well. However, the depth and flexibility of the tree likely introduce overfitting, a known behavior of unpruned decision trees.

---

## Summary

- **Progressed MLP Regressor (Regression):**  
    Increasing the number of hidden layers and training iterations improved the model’s performance, achieving an R² of 0.720 and reducing the MSE to 0.685. This indicates a stronger ability to capture variability in traffic volume compared to the original model, though with a potential risk of overfitting.

- **Progressed Logistic Regression (Classification):**  
    Despite scaling features and adjusting the regularization strength, the model’s accuracy remained at approximately 39.8%, with persistent misclassification across classes. The progressed logistic regression model confirmed the limitations of linear decision boundaries in modeling complex traffic patterns.

- **Progressed Decision Tree Classifier (Classification):**  
    Allowing the Decision Tree to grow without depth constraints significantly improved its performance, achieving an accuracy of 81.5%. Class-wise metrics were balanced, with F1-scores around 0.80, suggesting the model captured complex relationships in the data. However, the full tree growth also likely introduced overfitting.

---

# Midterm Report

---

# Model Comparison and Inferences

For this project, I applied three different models on the dataset, each targeting a specific problem formulation. The models evaluated were:

---

## MLP Regressor

A neural network-based regression model was employed to predict a continuous target variable. The key performance metrics for this model were:

- **R² Score:** 0.493  
  This indicates that the model explains about 49% of the variance in the target variable.

- **Mean Squared Error (MSE):** 1.243  
  On average, the squared error between the predicted and actual values is 1.243.

While the MLP Regressor captures a moderate amount of the data variability, there remains substantial unexplained variance.

---

## Logistic Regression

This linear classification model was evaluated with the following performance metrics:

- **Accuracy:** 39.8%  
  The model correctly classified about 40% of the instances.

- **Confusion Matrix and Classification Report:**  
  The confusion matrix shows significant misclassification across classes, and the precision and recall values (ranging roughly from 0.35 to 0.47 and 0.21 to 0.53, respectively) indicate that the model is struggling to correctly identify instances for each class.
 
Logistic regression, which relies on linear decision boundaries, seems insufficient to capture the complexity of the data’s class structure. Its lower overall accuracy and imbalanced performance across classes suggest that a more flexible or non-linear model might be more appropriate.

---

## Decision Tree Classifier

This model builds a tree-like structure to classify the data and is capable of modeling non-linear relationships. The performance metrics were:

- **Accuracy:** 63.7%  
  The decision tree correctly classified approximately 64% of the instances.

- **Confusion Matrix and Classification Report:**  
  With precision values ranging from 0.55 to 0.78 and recall values from 0.60 to 0.73 across classes, the decision tree provides a more balanced performance compared to logistic regression.
 
The decision tree classifier outperformed logistic regression by capturing more complex patterns in the data, leading to higher overall accuracy and better class-wise performance.

---

## Summary

- **MLP Regressor (Regression):**  
  Achieved a moderate performance with an R² of 0.493 and an MSE of 1.243. This indicates that while the model explains nearly half of the variance, there is room for improvement.

- **Logistic Regression (Classification):**  
  With an accuracy of about 40%, logistic regression struggled with the multi-class problem, likely due to its reliance on linear separability which does not sufficiently capture the data’s complexity.

- **Decision Tree Classifier (Classification):**  
  The best performer among the classification models with an accuracy of 63.7% and more balanced precision and recall scores across classes. Its ability to model non-linear relationships makes it a strong candidate for this dataset.
