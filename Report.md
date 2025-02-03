# Task 1 - Decision Tree based ID3 Algorithm
The **Iterative Dichotomiser 3 Decision Tree Algorithm** is a classification algorithm that builds a **decision tree** by choosing the attribute that provides the highest **Information Gain**. It follows a **greedy approach**, recursively selecting the best attribute for partitioning the dataset until an optimal tree is formed.It is prone to Overfitting,cannot handle Continuous Data directly and Biased towards attributes with many values(Gain Ratio solves this).

###  Steps   

1. Calculate Entropy of the Dataset – Measures impurity in the dataset.  H(S)=−∑(i=1,n)(pᵢ * log₂(pᵢ))
2. Calculate Information Gain for Each Feature – Determines the effectiveness of an attribute in splitting the data.  IG(S,A)=H(S)−H(S∣A)
3. Select the Feature with the Highest Information Gain** – This becomes the root node.  
4. Split the Dataset based on the selected feature and repeat the process recursively.
5. It is stopped when all examples belong to the same class,no more attributes remain to split or the dataset is empty.  
