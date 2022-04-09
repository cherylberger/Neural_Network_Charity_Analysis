# Neural_Network_Charity_Analysis
#### Module 19 Challenge
### Cheryl Berger

## Overview
 
From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.''

## Results 

### Deliverable 1: Preprocessing Data for a Neural Network Model
Using your knowledge of Pandas and the Scikit-Learn’s StandardScaler(), you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.

### Data Preprocessing
1. What variable(s) are considered the target(s) for your model?
2. What variable(s) are considered to be the features for your model?
3. What variable(s) are neither targets nor features, and should be removed from the input data?

- Create a density plot for column values
- Create bins for low counts
- Place rare categorical values in a separate column
- Create an instance of OneHotEncoder and fit the encoder with values
- Merge DataFrames and drop original columns
- Use the StandardScaler() module to standardize numerical variables
- Generate a categorical variable list
- Split the preprocessed data into features and target arrays, and scale the data

### Deliverable 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

### Compiling, Training, and Evaluating the Model
1. How many neurons, layers, and activation functions did you select for your neural network model, and why?
2. Were you able to achieve the target model performance?
3. What steps did you take to try and increase model performance?

### Coding steps
- Deep learning model design
- Train and evaluate the model
- Create a checkout and callback to save the model’s weights
- Save the results after training


### Deliverable 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

- Optimize a neural network
- Deep learning model design
- Train and evaluate the model
- Logistic vs neural network
- Support vector machine vs deep learning
- Random forest vs deep learning
- Create a checkout and callback to save the model’s weights
- Save the results after training

## Summary: 
Summarize the overall results of the deep learning model. 


Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
