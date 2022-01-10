import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from utils import haversine_vectorized


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, column_name, tz_name='America/New_York'):
        self.column_name = column_name
        self.tz_name = tz_name

    def fit(self, X, y=None):  
        return self

    def transform(self, X, y=None):
        tz = self.tz_name
        col = self.column_name
        df = X.copy()
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        df[col] = df[col].dt.tz_convert(tz)
        df['hour'] = df[col].dt.hour
        df['weekday'] = df[col].dt.dayofweek
        df['month'] = df[col].dt.month
        df['year'] = df[col].dt.year
        X_ = df[[col, 'hour', 'weekday', 'month', 'year']].set_index(col).copy()
        return X_

class DistanceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, start_lat="pickup_latitude", start_lon="pickup_longitude",
                 end_lat="dropoff_latitude", end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = pd.DataFrame(haversine_vectorized(X)).rename(columns={0: "distance"}).copy()
        return X_[['distance']]