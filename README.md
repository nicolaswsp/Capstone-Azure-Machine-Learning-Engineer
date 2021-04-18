*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

This project aims to demonstrate how to train a model using Azure Machine Learning Studio and put it in production to an end user. The dataset used was Credit Card Fraud Detection dataset provided in the Kaggle website https://www.kaggle.com/mlg-ulb/creditcardfraud, where the goal is to predict whether a banking transaction would result in fraud based on several features. The training process was made using Auto ML and also Hypervisor through Azure Machine Learning Studio. After that, it was chosen the best model to accordingly to the best algorithm, VotingEnsemble, and deployed into production. With the deployed model, it was enabled the Application Insight using Azure Python SDK. For the documentation, it was used Swagger instance running with the documentation for the HTTP API of the model. The model was consumed using the deployed scoring URI and key authentication running the endpoint.py script against the API producing JSON output from the model. I was used Apache Benchmark to retrieve the performance result details running against the HTTP API. 

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.
The dataset used was Credit Card Fraud Detection dataset provided in the Kaggle website https://www.kaggle.com/mlg-ulb/creditcardfraud. The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. It contains only numerical input variables which are the result of a PCA transformation.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
This project main objective is to predict fraud in credit card baking transactions help credit card companies to be able to recognize fraudant transaction to avoid future purcharses that were not requested from them. The datatset consists in 30 features containing only numerical input variables which are the result of a PCA transformation to predict the feature 'Class' that is the response variable and it takes value 1 in case of fraud and 0 otherwise.

### Access
*TODO*: Explain how you are accessing the data in your workspace.
This dataset was downloaded, uploaded and created a dataset with it inside the Azure Machine Learning Studio where the dataset can be consumed through the workspace configuration and the dataset name. 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
Automated Machine Learning was used to train the model in the Azure Machine Learning Studio. For this task the Auto ML main configuration consists in binary classification, for the since primary metric was chosen AUC weighted since it works well in imbalanced datasets, and the featurization was chosen to be automatic.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
For the hyperparameter tuning using Azure Hypervisor was chosen to use and ensemble model through the Scikit-Learn library, using a random forest classifier.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
