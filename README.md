# Neural_Network_Charity_Analysis
#### Module 19 Challenge
### Cheryl Berger

## Overview
 
From Alphabet Soup’s business team, Beks received a CSV file ++++++++ containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

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

The purpose of this analysis is to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

## Results 

### Deliverable 1: Preprocessing Data for a Neural Network Model
Using Pandas and the Scikit-Learn’s StandardScaler(), preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.

Import dependencies and load the datafile as indicated in the image of the code block below:
![image](https://user-images.githubusercontent.com/94234511/162554578-40c3c3f5-a7a7-45c6-bddd-290a159e00fe.png)

#### Data Preprocessing
Follow these basic steps to preprocess the data for compilation and modeling. 
- Create a density plot for column values
- Create bins for low counts
The variables APPLICATION_TYPE and CLASSIFICATION were identified for binning.  See the code snippets below:
 ![image](https://user-images.githubusercontent.com/94234511/162554757-00442e54-8878-40d9-8563-27b877bae39e.png)

 ![image](https://user-images.githubusercontent.com/94234511/162554792-a16298a5-9838-47e6-b104-f8f7091ae417.png)

 ![image](https://user-images.githubusercontent.com/94234511/162554771-f877cd4e-99ce-4571-9e62-bb7173777ce9.png)

 ![image](https://user-images.githubusercontent.com/94234511/162554804-a88f3d78-802e-40d9-a2db-0f657142e2b7.png)

- Place rare categorical values in a separate column
- Create an instance of OneHotEncoder and fit the encoder with values
- Merge DataFrames and drop original columns
- Use the StandardScaler() module to standardize numerical variables
- Generate a categorical variable list
- Split the preprocessed data into features and target arrays, and scale the data
- Split the data into training and test sets

#### 1. What variable(s) are considered the target(s) for your model?
The variable IS_SUCCESSFUL contains the column with the target for the model as we seek to predict the success of applicants from the data provided. 
![image](https://user-images.githubusercontent.com/94234511/162554845-9653092e-a3a7-4df6-ae27-b6b3df39aec4.png)
 
#### 2. What variable(s) are considered to be the features for your model?
After binning, the variables APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT and SPECIAL_CONSIDERATIONS have a reasonable number of bins and maybe useful in developing the predictive model. We would consider using these features in our model design and assess the predictive performance in Deliverable 2. See the code images below to generate a categorical list and use One Hot Encoder method to fit and transform the categories and add th encoded variables to a new dataframe. 

![image](https://user-images.githubusercontent.com/94234511/162554122-9b0674fc-7f6c-4037-a16b-a5c0319f07ab.png)

![image](https://user-images.githubusercontent.com/94234511/162554180-b3919dcb-3d87-480d-b137-d61f1cd39160.png)

#### 3. What variable(s) are neither targets nor features, and should be removed from the input data?
The variables containing identification are not useful in predicting and can confuse the model if the datatype is integer. For this exercise we removed the 'EIN' and 'NAME' columns as indicated in the code below.
![image](https://user-images.githubusercontent.com/94234511/162554156-889de27d-a629-444e-8c3e-cac58e8e6f98.png)

- Split the preprocessed data into features and target arrays, and scale the data
![image](https://user-images.githubusercontent.com/94234511/162554216-c575b52c-5c51-4247-9a1b-5917640bddac.png)

### Deliverable 2: Compile, Train, and Evaluate the Model
Using TensorFlow, design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Then, compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy and save the results after training.  

#### Coding steps
Follow these basic steps to perform the inital data compilation and modeling using the preprocessed data from Deliverable 1.

- Deep learning model design
  ![image](https://user-images.githubusercontent.com/94234511/162555389-bd642ccd-0b26-4b53-817e-ee4ef7e1d476.png)
  
- Create a checkout and callback to save the model’s weights
  ![image](https://user-images.githubusercontent.com/94234511/162555569-b90eb51f-48d7-4e72-be01-ea3340f05fe0.png)

- Train and evaluate the model
  ![image](https://user-images.githubusercontent.com/94234511/162555438-33cf4965-5043-45b7-8a29-ad842bd0f06f.png)

- Save the results after training
  ![image](https://user-images.githubusercontent.com/94234511/162555593-52e6a3dd-1cbc-49eb-acbd-50f8d406d96f.png)

#### Compiling, Training, and Evaluating the Model
1. How many neurons, layers, and activation functions did you select for your neural network model, and why?
There were a totall of 110 neurons deployed in 2 hidden layers.  The relu activitation function was used for both hidden layers and the sigmoid function was used for the output layer. 

2. Were you able to achieve the target model performance?
The training accuracy was just slightly higher than 50% (53.24) but the model was only able to give 69% accuracy with the test data.  Test loss was 1.399.  Training accuracy peaked at about EPOCH 30 (>67%) but started to decline after about EPOCH 42 suggesting some overfitting.  

### Deliverable 3: Optimize the Model
Using TensorFlow, optimize the model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

- Optimize a neural network
- Deep learning model design
- Train and evaluate the model
- Logistic vs neural network
- Support vector machine vs deep learning
- Random forest vs deep learning
- Create a checkout and callback to save the model’s weights
- Save the results after training

3. What steps did you take to try and increase model performance?

#### Attempt #1 
- Binned the ASK_AMT Column into smaller bins and added it to the categorical list for encoding 
 ![image](https://user-images.githubusercontent.com/94234511/162556112-b1783c1e-4b59-4d76-9d99-58d4c766bf60.png)

 ![image](https://user-images.githubusercontent.com/94234511/162556104-3443f2f0-c8ac-4c5c-a739-102405f81d5a.png)

 ![image](https://user-images.githubusercontent.com/94234511/162556176-1ff1dadd-b4ea-42cb-b994-5d399c6fcfb4.png)

- Once the new model is compiled, scaled and analyzed, the accuracy of the model improved to 74.45% for the training dataset and 73% for the test data  
![image](https://user-images.githubusercontent.com/94234511/162556835-c63c7468-a359-4647-a0ec-e6db97348dc7.png)

#### Attempt #2
- Add additional layers to the neural network

  ![image](https://user-images.githubusercontent.com/94234511/162556513-77e34448-c32a-4e51-949d-06d1ef6b4c99.png)

- Once the new model is compiled, scaled and analyzed, the accuracy of the model is largely unchanged for both the training and test datasets at 73%, just a bit below the target. A quick comparison of the accuracy during fitting shows a fairly small rate of change suggesting no added value from additional training (increasing the # number of EPOCHS).  

  ![image](https://user-images.githubusercontent.com/94234511/162556977-db1bc9b4-508e-446c-b986-62c3addff155.png)

  ![image](https://user-images.githubusercontent.com/94234511/162556990-6c6594ba-a48d-4b82-81d1-3b8f0c5c2d4b.png)

#### Attempt #3

- A return to the dataset for a second look at the features may be the only way to further improve the models ability to reach the goal of 75% accuracy to predict successful applicants. 



## Summary: 
Summarize the overall results of the deep learning model. 

As shown above, the initial setup for the model did not perform at the required level, coming in no better than 69%. After binning the ASK_AMT column (which has noisy data) the model design was iterated and obtained a much improved accuracy rating of 73% for the test data.  In an effort to realize the goal of 75% accuracy, a second attempt to improve the model was performed by adding additional layers to the neural network.  However, one the data was compiled, scaled and analyzed, the accuracy was improved over the initial model but still no greater than 73% for the test data.  

All other models had lower final accuracies than this, 
It an be seen that changing the number of hidden layers and neurons had a negligible effect on increasing model accuracy. Further adjustment may lead to an invalid model or one that overfits the dataset.

I would recommend a Random Forest classifier for an alternate model design. This is due to Random Forest ability of performing binary classification, the ability to handle large datasets, and the reduction in code which can achieve comparable accuracy predictions.  Alternatively, re-evaluating the data used in the model may provide insights into other features that could be analyzed. 

