import os
import sys
from src.exception import CustomException 
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data:str = os.path.join("artifacts","data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info("Data Ingestion is started")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Reading the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path))
            df.to_csv(self.ingestion_config.raw_data)

            logging.info("Train test split initiated")
            train,test = train_test_split(df,test_size=0.2,random_state=42)

            train.to_csv(self.ingestion_config.train_data_path)
            test.to_csv(self.ingestion_config.test_data_path)
            logging.info("Ingestion of data is completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                    )

        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)


a = DataIngestion()
a.initiate_ingestion()


