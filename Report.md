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

 ![](https://raw.githubusercontent.com/ashith-17/Marvel-level-02/refs/heads/main/pics/GBM%20For%20Regression.png)




     


