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

The AutoConfig configuration in the same Notebook followed these steps:

* Created a dataset from the provided Kaggle dataset uploading it on Azure ML Studio
* Splitted data into train and test sets
* Modified the AutoML config
* Submitted the AutoML run
* Saved the best model
* Deployed the best model
* Consumed the best model

The AutoML config was the following parameters:
* experiment_timeout_minutes=20  
  This is an exit criterion when the experiment time exceeds the time out established. This avoids to time out when occurs failures.
* task='classification'  
  This sets the type of experiment that is set to run, in this case was a binary classification.
* primary_metric='AUC_weighted'  
  This sets the primary metric that is used as parameters to choose the best model.
* label_column_name='Class'  
  This criterion sets the dataset collumn that is aimed to be predicted.
* enable_early_stopping=True  
  This sets to enable early termination if the score is not improving in the short term.
* featurization='auto'  
  This is an indicator for whether featurization step should be done automatically or not. 



### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The best performing model was the VotingEnsemble among the several tested. Some of these models were LightGBM, Xgboost, ExtremeRandomTrees, StandardScalerWrapper, RandomForest. A voting ensemble works by combining the predictions from multiple models. It uses Soft Voting that predicts the class with the largest summed probability from models. The AutoML generates parameters of its inner estimators. It can be highlighted that the VotingEnsemble model used L1 Regularization as one of these parameters. The L1 Regularization adds "absolute value of magnitude" of coefficient as penalty term to the loss function shrinking the less important feature’s coefficient to zero thus, removing some feature altogether.
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
