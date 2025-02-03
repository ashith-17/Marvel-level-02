# Task 1 - Decision Tree based ID3 Algorithm
The **Iterative Dichotomiser 3 Decision Tree Algorithm** is a classification algorithm that builds a **decision tree** by choosing the attribute that provides the highest **Information Gain**. It follows a **greedy approach**, recursively selecting the best attribute for partitioning the dataset until an optimal tree is formed.It is prone to Overfitting,cannot handle Continuous Data directly and Biased towards attributes with many values(Gain Ratio solves this).

###  Steps in ID3

1. Calculate Entropy of the Dataset – Measures impurity in the dataset.  H(S)=−∑(i=1,n)(pᵢ * log₂(pᵢ))
2. Calculate Information Gain for Each Feature – Determines the effectiveness of an attribute in splitting the data.  IG(S,A)=H(S)−H(S∣A)
3. Select the Feature with the Highest Information Gain** – This becomes the root node.  
4. Split the Dataset based on the selected feature and repeat the process recursively.
5. It is stopped when all examples belong to the same class,no more attributes remain to split or the dataset is empty.  
[Implementation of ID3](https://www.kaggle.com/code/ashith1709/id3-algorithm)
![](https://raw.githubusercontent.com/ashith-17/Marvel-level-02/refs/heads/main/pics/Id3%20algorithm%20DT.png)


# Task 2 - Naive Bayesian Classifier

The **Naive Bayes classifier** is a probabilistic model based on **Bayes' Theorem** with an assumption of feature independence.  
Equation of Bayes Theorem.P(C|X) = P(X|C) * P(C) / P(X). Types of Naive Bayes -

1. **Gaussian Naive Bayes**:
     Assumes features follow a **Gaussian (normal) distribution** and used for **continuous data**.

2. **Multinomial Naive Bayes**:
     Assumes features follow a **multinomial distribution** (e.g., word counts) and used for **discrete data** like text.

3. **Bernoulli Naive Bayes**:
     Assumes binary features (0 or 1) and used for binary attributes.

---

### Steps in Naive Bayes

1. **Calculate Prior Probabilities**:  
   Compute P(C), the probability of each class.
   
2. **Calculate Likelihood**:  
   Compute P(X|C) for each feature using probability distribution function(∏ P(xᵢ|C) ).
   
3. **Apply Bayes' Theorem**:  
   Compute P(C|X) for each class.

4. **Predict Class**:  
   Choose the class with the highest posterior probability.

   [Implementing it for text classification](https://www.kaggle.com/code/ashith1709/notebook045b1da53d)
   


