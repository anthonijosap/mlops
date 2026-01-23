import os
import sys  
import pandas as pd
import numpy as np
from src.exception import CustomException

def save_object(file_path, obj):
    import pickle
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)