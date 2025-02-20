# Task 1 - Decision Tree based ID3 Algorithm
The **Iterative Dichotomiser 3 Decision Tree Algorithm** is a classification algorithm that builds a **decision tree** by choosing the attribute that provides the highest **Information Gain**. It follows a **greedy approach**, recursively selecting the best attribute for partitioning the dataset until an optimal tree is formed.It is prone to Overfitting,cannot handle Continuous Data directly and Biased towards attributes with many values(Gain Ratio solves this).

 Steps in ID3

1. Calculate Entropy of the Dataset – Measures impurity in the dataset.  H(S)=−∑(i=1,n)(pᵢ * log₂(pᵢ))
2. Calculate Information Gain for Each Feature – Determines the effectiveness of an attribute in splitting the data.  IG(S,A)=H(S)−H(S∣A)
3. Select the Feature with the Highest Information Gain.This becomes the root node.
4. Split the Dataset based on the selected feature and repeat the process recursively.
5. It is stopped when all examples belong to the same class,no more attributes remain to split or the dataset is empty.  
[Implementation of ID3](https://www.kaggle.com/code/ashith1709/id3-algorithm)
![](https://raw.githubusercontent.com/ashith-17/Marvel-level-02/refs/heads/main/pics/Id3%20algorithm%20DT.png)


# Task 2 - Naive Bayesian Classifier

The **Naive Bayes classifier** is a probabilistic model based on **Bayes' Theorem** with an assumption of feature independence.  
Equation of Bayes Theorem.P(C|X) = P(X|C) * P(C) / P(X). Types of Naive Bayes -  
  
  1.Gaussian Naive Bayes: Assumes features follow a Gaussian (normal) distribution and used for continuous data.  
  2.Multinomial Naive Bayes: Assumes features follow a multinomial distribution (e.g., word counts) and used for discrete data like text.  
  3.Bernoulli Naive Bayes:  Assumes binary features (0 or 1) and used for binary attributes.  

 Steps in Naive Bayes:  
 1.Calculate Prior Probabilities:Compute P(C), the probability of each class.  
 2.Calculate Likelihood:Compute P(X|C) for each feature using probability distribution function(∏ P(xᵢ|C) ).  
 3.Apply Bayes' Theorem:Compute P(C|X) for each class.  
 4.Predict Class:Choose the class with the highest posterior probability.  
   
   ![](https://raw.githubusercontent.com/ashith-17/Marvel-level-02/refs/heads/main/pics/Prob%20df.jpg)
   
   [Implementing it for text classification](https://www.kaggle.com/code/ashith1709/notebook045b1da53d)



# Task 3 - Ensemble techniques


Ensemble learning is a technique that combines multiple models to improve performance, accuracy, and robustness. By aggregating the predictions of several models, ensemble methods can often outperform individual models. The key techniques inlcude:

 1. Bagging (Bootstrap Aggregating)
   A method to reduce variance by training multiple models on random subsets of data and averaging their predictions.Example: Random Forest.
 2. Boosting
    An iterative technique that focuses on correcting errors by giving more weight to misclassified data points in each subsequent model.Example: AdaBoost, Gradient Boosting, XGBoost.
 3. Stacking
    Combines different models (base learners) and uses a meta-model to combine their predictions, leveraging the strengths of multiple algorithms.Example: Stacked Generalization.

![](https://raw.githubusercontent.com/ashith-17/Marvel-level-02/refs/heads/main/pics/Ensemble%20techniques.jpg)  

[Basic Ensemble techniques](https://www.kaggle.com/code/ashith1709/ensemble-techniques)   

[Ensemble techniques on Titanic Dataset](https://www.kaggle.com/code/ashith1709/ensemble-on-titanic)  


# Task 4 - Random Forest, GBM and Xgboost  

### Random Forest:

 1.Bootstrapping:  
 The algorithm randomly samples data with replacement from the original dataset.Each sample is used to train an independent decision tree.  

 2.Feature Randomness & Decision Tree Splitting:  
 At each node split, a random subset of features is selected instead of considering all features.The best feature is chosen based on: Gini Impurity/Entropy(for classification) 
 and Mean Squared Error (for regression).This feature randomness reduces correlation between trees, improving diversity and preventing overfitting.

 3.Training Multiple Decision Trees:  
 Each decision tree is trained independently on a different bootstrap sample.The trees grow to their maximum depth unless a stopping condition is met.Since each tree is trained on different data and features, they make diverse predictions.

 4.Aggregation of Predictions:  
 For Classification:Each tree votes for a class, and the majority class is selected.
 For Regression: Predictions from all trees are averaged to obtain the final result.

 5.Model Evaluation & Feature Importance:  
**Out-of-Bag (OOB) error** is calculated using data left out during bootstrap sampling.Feature importance is determined based on:Gini Importance and Permutation Importance.
  Hyperparameters like the number of trees, depth, and number of features per split are optimized for better performance.  

  [Implementing Random Forest](https://www.kaggle.com/code/ashith1709/random-forest)  

 ### Gradient Boost Machine: 

### Gradient Boosting Machine (GBM) for Regression: 

 ![](https://raw.githubusercontent.com/ashith-17/Marvel-level-02/refs/heads/main/pics/GBM%20For%20Regression.png) 

1.Initialize the Model:  
The model starts with a constant function,typically the mean of the target variable 𝑦.The purpose of this step is to establish an initial model that minimizes the loss function(MSE) for the given data.

2.Compute Residuals (Errors):  
 Calculate the difference between actual values and the current model’s predictions.These residuals represent the errors that the next model will learn to correct.

3.Train Weak Learners (Decision Trees):    
A shallow Decision Tree (weak learner) is trained on the residuals. Instead of predicting the target variable, it learns to predict the residual errors.The number of features considered at each split is often limited to prevent overfitting.

4.Update the Model Using Learning Rate:  
The new model is added to the existing model with a scaling factor called the **learning rate (η)**.

5.Repeat Until Convergence:  
 Steps 2 to 4 are repeated for a predefined number of iterations (trees) or until the improvement in error is minimal.The final prediction is obtained by summing the contributions from all weak learners.  

[GBM For Regression](https://www.kaggle.com/code/ashith1709/gbm-for-regression)

### Gradient Boosting Machine (GBM) for Classification:  

 1.Initialize the Model with a Constant Value: 
The model starts with an initial prediction that minimizes the loss function.For **binary classification**, this is typically the log-odds of the positive class:  
   F_0(x) = log(p/1-p),where p is the proportion of positive class instances.  


2.Iterate Over M Weak Learners (Boosting Steps): 
For each boosting iteration m, we improve the model by adding a new tree.  

---  

2.1: Compute Pseudo-Residuals:   
Instead of simple residuals, we compute **pseudo-residuals** using the gradient of the **log loss function**:  
  r_{im} = y_i - p_i
  where p_i is the predicted probability for class 1.  
  
 
2.2: Train a Weak Learner on Pseudo-Residuals:  
 We fit a **decision tree** to predict the computed pseudo-residuals.The purpose of this step is for the tree to learn patterns in the residuals and improve predictions.  


2.3 :Compute the Scaling Factor:  
 The optimal scaling factor controls how much the new weak learner contributes to the model.It is found by minimizing the loss function.The purpose of this step is to adjust the contribution of the weak learner to prevent overfitting.  

 2.4: Update the Model: 
 The model is updated by adding the new weak learner multiplied by the scaling factor:  
  F_m(x) = F_{m-1}(x) + gamma_m h_m(x)  
The predicted probabilities are then updated using the **sigmoid function** for binary classification.  

  ---
 
3: Final Prediction:
After M iterations, the final model is obtained F_M(x).The final prediction for binary classification is:  
y={1,p_i>=0.5  and  0,p_i<0.5 }  

[GBM For Classification](https://www.kaggle.com/code/ashith1709/gbm-for-classification)  

### XGBOOST (Extreme Gradient Boosting)  
 
![](https://raw.githubusercontent.com/ashith-17/Marvel-level-02/refs/heads/main/pics/XGBOOST.jpg)   

This breakdown highlights the step-by-step differences between regression and classification in XGBoost, without formulas—just the key terms.  

1. Initial Prediction  
Regression:Mean of the target variable as the initial prediction.Classification:Log-odds of the positive class probability as the initial prediction, converts log-odds to probability using the sigmoid function.  

2. Compute Residuals (Gradients of the Loss Function)  
Regression:Gradients as the difference between actual and predicted values.Classification:Gradients as the difference between actual labels and predicted probabilities, uses sigmoid-transformed probabilities in the calculation.  

3. Compute Hessians (Second-Order Derivative of the Loss Function)  
Regression:Hessians are set to 1, since MSE has a constant second derivative.Classification:Hessians depend on predicted probabilities, adjusting based on how confident the model is.  

4. Build the Decision Tree (Finding Best Splits)  
Regression:Gain function determines the best splits based on gradients and Hessians.Classification:Gain function is used, but Hessians are different since they depend on probabilities.  

5. Compute Leaf Weights  
Regression:Leaf weights are computed using gradients and Hessians, Hessians remain 1 throughout.Classification:Leaf weights are computed using gradients and Hessians, Hessians are probability-dependent, making weight calculations different.  

6. Update Predictions  
Regression:Predictions are updated by adding scaled leaf weights using the learning rate.Classification:Predictions are updated by adding scaled leaf weights using the learning rate, predictions are converted into probabilities using the sigmoid function.  

7. Stop When Convergence is Reached  
Regression:Training stops when stopping criteria are met, such as minimum improvement in MSE loss, maximum number of trees, or early stopping.Classification:Training stops when stopping criteria are met, such as minimum improvement in log-loss, maximum number of trees, or early stopping.

[XGB For cla](https://www.kaggle.com/code/ashith1709/xgboost-for-classification)
[XGB For Reg](https://www.kaggle.com/code/ashith1709/xgboost-for-regression)
  





     


