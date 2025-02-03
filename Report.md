# ID3 (Iterative Dichotomiser 3) Decision Tree Algorithm  

The **ID3 algorithm** is a classification algorithm that builds a **decision tree** by choosing the attribute that provides the highest **Information Gain**. It follows a **greedy approach**, recursively selecting the best attribute for partitioning the dataset until an optimal tree is formed.  

## Key Steps in ID3  

1. **Calculate Entropy of the Dataset** – Measures impurity in the dataset.  
2. **Calculate Information Gain for Each Feature** – Determines the effectiveness of an attribute in splitting the data.  
3. **Select the Feature with the Highest Information Gain** – This becomes the root node.  
4. **Split the Dataset** based on the selected feature and repeat the process recursively.  

### Stopping Conditions:  
- All examples belong to the same class.  
- No more attributes remain to split.  
- The dataset is empty.  
