import os
import sys  
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    import pickle
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    model_report = {}
    try:
        for model_name, model in models.items():
            params = params.get(model_name, {})

            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            
            model.fit(X_train, y_train)


            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[model_name] = r2
        return model_report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    import pickle
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)