
import os
import sys
from src.logger.logging import logging
from src.exception.exception import CustomException
import pandas as pd
import mlflow


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':

    name = sys.argv[1]
    # mlflow.set_tracking("http://localhost:5000")
    mlflow.set_experiment(name)

    with mlflow.start_run() as run:

        obj=DataIngestion()
        train_data_path,test_data_path=obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr,test_arr,_,col_names,Data_Transformation_tags,preprocessing_obj, preprocessor_path=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
        # with mlflow.start_run(nested=True):
        res = 0
        for key in col_names:
            for i in col_names[key]:
                res += 1
                log = key + str(res)
                mlflow.log_param(log, i)
        
        for steps in Data_Transformation_tags:
            for step in Data_Transformation_tags[steps]:
                ste = steps + '-' + step[0]
                mlflow.log_param(ste, step[1])

        confusion_matrix_path = None
        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, "confusion_materix")
        
        roc_auc_plot_path = None
        if not roc_auc_plot_path == None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")

        mlflow.log_artifact(preprocessor_path,'preprocessing_obj')
        
        model_trainer=ModelTrainer()
        model_report,best_model_score,best_model_name,best_model_path, metrics = model_trainer.initate_model_training(train_arr,test_arr)
        
        for model_name, score in model_report.items():
            mlflow.log_metric(model_name, score)

        for model_name, score in metrics.items():
            mlflow.log_metric(model_name, score)
        
        mlflow.log_metric('best_model_'+best_model_name, best_model_score)

        mlflow.log_artifact(best_model_path, 'best_model_'+best_model_name)



