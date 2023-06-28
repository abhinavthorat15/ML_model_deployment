import sys
import os
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from src.utils import save_output


@dataclass
class DataTransformationConfig:
    preprocess_pipeline_path:str = os.path.join("artifacts","preprocess.pkl")


class Datatransformation:
    def __init__(self):
        self.pipeline_path = DataTransformationConfig()
        
    def transformation_pipeline(self):
        logging.info("Variable Transformation Pipeline triggered")
        try:

            num_variables = ['reading_score', 'writing_score']
            cat_variables = ['gender', 
                            'race_ethnicity', 
                            'parental_level_of_education', 
                            'lunch',
                            'test_preparation_course'
                            ]   

            num_pipeline = Pipeline(
                steps=[
                    ("imputation",SimpleImputer(strategy="median")),
                    ("scaling",StandardScaler(with_mean= False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputation",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder()),
                    ("scaling",StandardScaler(with_mean=False))
                ]
            )
        
            master_pipeline = ColumnTransformer(
                [
                    ("numeric_transformation",num_pipeline,num_variables),
                    ("cat_transformation",cat_pipeline,cat_variables)
                ]
            )

            return master_pipeline
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_transformation(self,train_path:str,test_path:str):
        try:
            train_data = pd.read_csv(train_path)
            test_data  = pd.read_csv(test_path)
            logging.info("Train and Test data has been imported as a dataframe")

            master_pipeline = self.transformation_pipeline()

            target_column = "math_score"
            input_train_data_features = train_data.drop(columns = [target_column],axis=1)
            input_test_data_features = test_data.drop(columns = [target_column],axis=1)

            train_target_column = train_data[target_column]
            test_target_column = test_data[target_column]

            input_train_data_features_arr = master_pipeline.fit_transform(train_data)
            input_test_data_features_arr = master_pipeline.transform(test_data)

            train_array = np.c_[input_train_data_features_arr,np.array(train_target_column)]
            test_array = np.c_[input_test_data_features_arr,np.array(test_target_column)]

            save_output(
                file_path=self.pipeline_path.preprocess_pipeline_path,
                obj = master_pipeline
            )
            logging.info("The train and test dataset have been transformed")
            logging.debug("The preprocessing pipeline has been executed")
            return (
                train_array,
                test_array,
                self.pipeline_path.preprocess_pipeline_path
                
            )

        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)
        

