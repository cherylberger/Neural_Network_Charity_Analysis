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
Using your knowledge of Pandas and the Scikit-Learn’s StandardScaler(), you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2. Import dependencies and load the datafile as indicated in the image of the code block below:
![image](https://user-images.githubusercontent.com/94234511/162554578-40c3c3f5-a7a7-45c6-bddd-290a159e00fe.png)

### Data Preprocessing
Follow these basic steps to preprocess the data for compilation and modeling. 
- Create a density plot for column values
- Create bins for low counts
- Place rare categorical values in a separate column
- Create an instance of OneHotEncoder and fit the encoder with values
- Merge DataFrames and drop original columns
- Use the StandardScaler() module to standardize numerical variables
- Generate a categorical variable list
- Split the preprocessed data into features and target arrays, and scale the data

#### 1. What variable(s) are considered the target(s) for your model?
The variable IS_SUCCESSFUL contains the column with the target for the model as we seek to predict the success of applicants from the data provided. 
   # Split our preprocessed data into our features and target arrays
   y = application_df.IS_SUCCESSFUL
   X = application_df.drop("IS_SUCCESSFUL", axis=1)

#### 2. What variable(s) are considered to be the features for your model?
After binning, the variables APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT and SPECIAL_CONSIDERATIONS have a reasonable number of bins and maybe useful in developing the predictive model. We would consider using these features in our model design and assess the predictive performance in Deliverable 2. See the code images below to generate a categorical list and use One Hot Encoder method to fit and transform the categories and add th encoded variables to a new dataframe. 

![image](https://user-images.githubusercontent.com/94234511/162554122-9b0674fc-7f6c-4037-a16b-a5c0319f07ab.png)

![image](https://user-images.githubusercontent.com/94234511/162554180-b3919dcb-3d87-480d-b137-d61f1cd39160.png)

#### 3. What variable(s) are neither targets nor features, and should be removed from the input data?
The variables containing identification are not useful in predicting and can confuse the model if the datatype is integer. For this exercise we removed the 'EIN' and 'NAME' columns as indicated in the code below.
![image](https://user-images.githubusercontent.com/94234511/162554156-889de27d-a629-444e-8c3e-cac58e8e6f98.png)

![image](https://user-images.githubusercontent.com/94234511/162554216-c575b52c-5c51-4247-9a1b-5917640bddac.png)

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
