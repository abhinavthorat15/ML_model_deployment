import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class predictpipeline:
    def __init__(self):
        pass

    def predict(self,dataframe):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path  = "artifacts\preprocess.pkl"
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            process_df = preprocessor.transform(dataframe)
            prediction = model.predict(process_df)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self): 
        try :
            custom_data_dict = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
             }
            dataframe = pd.DataFrame(custom_data_dict)
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)





