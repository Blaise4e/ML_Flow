import pandas as pd
import numpy

from utils import compute_rmse
from data import get_data, clean_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from encoders import DistanceTransformer, TimeFeaturesEncoder
from memoized_property import memoized_property

# The `Trainer` class is the main class. It should have:
# - an `__init__` method called when the class is instanciated
# - a `set_pipeline` method that builds the pipeline
# - a `run` method that trains the pipeline
# - an `evaluate` method evaluating the model

class Trainer():
    def __init__(self, X,y):
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('metering', DistanceTransformer()),
        ('scaler', StandardScaler())])
        time_pipe = Pipeline([('time_features', TimeFeaturesEncoder('pickup_datetime')),
        ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']),
        ('time', time_pipe, ['pickup_datetime'])])
        pipe = Pipeline([('preprocessor', preproc_pipe),
        ('regressor', LinearRegression())])
        return pipe

    def run(self,X, y, pipeline):
        '''returns a trained pipelined model'''
        # pipeline = self.set_pipeline()
        pipeline = pipeline.fit(X, y)
        return pipeline

    def evaluate(self,X, y, pipeline):
        '''returns the value of the RMSE'''
        prediction = pipeline.predict(X)
        rmse = compute_rmse(prediction, y)
        print(rmse)
        return rmse
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



file_path = "C:\\Dev_IA\\projets\\Brief15_MLOps\\TaxiFareModel\\data\\train.csv"

data = get_data(file_path,1000)
data = clean_data(data)
X = data.drop("fare_amount", axis = 1)
y = data["fare_amount"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

x = Trainer(X_train,y_train)
# print(x.df)
pipeline = Trainer(X_train,y_train).set_pipeline()
# print(pipeline)

pipeline = x.run(X_train, y_train, pipeline)
x.evaluate(X_test, y_test, pipeline)
