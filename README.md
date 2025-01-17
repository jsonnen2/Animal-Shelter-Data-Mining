Application of my Data Mining techniques to data from the Austin Animal Center. This dataset describes the outcomes for 26,000 cats who have lived here. My work focuses on uncovering 
relationships between variables and the cat's outcome. My first question was "Are cats with names more likely to be adopted?" I use Association Rule Mining to validate my relationships.
This technique fits a Random Forest Classifier to the data, then uses the depth in the tree for a decision split as a statistic to describe a rule's performance. Rules which are 
consistently placed higher in the Decision Tree are more important. I use that statistic to motivate the importance of a rule. I hope this work will culminate in a general purpose
tool in discovering relationships within a dataset.
