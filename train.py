from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import recall_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
import requests
from zipfile import ZipFile
from io import BytesIO
from azureml.core import Workspace, Dataset

subscription_id = '510b94ba-e453-4417-988b-fbdc37b55ca7'
resource_group = 'aml-quickstarts-142871'
workspace_name = 'quick-starts-ws-142871'

url = 'https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c534768_creditcardfraud/creditcardfraud.zip'
filename = requests.get(url).content
zf = ZipFile(BytesIO(filename), 'r')
zf.extractall()
transactions_df = pd.read_csv('./creditcard.csv')

# create the features dataframe
x = transactions_df.iloc[:, :-1]
# create the data label
y = transactions_df['Class']

# TODO: Split data into id train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=50, shuffle=True)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest")
    parser.add_argument('--max_depth', type=int, default=8, help="The maximum depth of the tree")
    parser.add_argument('--min_samples_split', type=int, default=2, help="The minimum number of samples required to split an internal node")
    parser.add_argument('--max_features', type=int, default=15, help="The number of features to consider when looking for the best split")

    args = parser.parse_args()

    run.log("Number of trees:", np.int(args.n_estimators))
    run.log("Maximum depth:", np.int(args.max_depth))
    run.log("Minimum number of samples:", np.int(args.min_samples_split))
    run.log("Maximum features:", np.int(args.max_features))


    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
    min_samples_split=args.min_samples_split, max_features=args.max_features).fit(x_train, y_train)

    y_pred = model.predict(x_test)
    recall = recall_score(y_test, y_pred, average='weighted')
    run.log("AUC_weighted", np.float(recall))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
