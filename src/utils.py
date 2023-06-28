import os
import pickle
from sklearn.pipeline import Pipeline
from src.exception import CustomException


def save_output(file_path:str,obj:Pipeline):
    try:
        directory_name = os.path.dirname(file_path)
        os.makedirs(directory_name,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException
