# source: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python
import pickle
import json
import pandas as pd
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    
    model_path = Model.get_model_path('best_automl_model')
    model = joblib.load(model_path)


def run(data):
    try:
        data = json.loads(data)
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error