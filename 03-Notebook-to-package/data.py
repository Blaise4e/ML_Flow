import pandas as pd
import numpy as np
file_path = "C:\\Dev_IA\\projets\\Brief15_MLOps\\TaxiFareModel\\data\\train.csv"

def get_data(file_path,nrows=1000):
    df = pd.read_csv(file_path, sep=',', nrows=nrows)
    return df

def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    df = df.drop(["key"], axis=1)
    df = df.dropna(how='any', axis='rows')
    df = df[(df.fare_amount > 0) & (df.fare_amount < 70)]
    df = df[(df.passenger_count < 8) & (df.passenger_count > 0)]
    df = df[df["pickup_latitude"].between(left = 40, right = 42 )]
    df = df[df["pickup_longitude"].between(left = -74.3, right = -72.9 )]
    df = df[df["dropoff_latitude"].between(left = 40, right = 42 )]
    df = df[df["dropoff_longitude"].between(left = -74, right = -72.9 )]
    return df

# file_path = "C:\\Dev_IA\\projets\\Brief15_MLOps\\TaxiFareModel\\data\\train.csv"
# x = get_data(file_path)
# x = clean_data(x)
# print(x.shape)

# df = Data_base()
# x = df.get_data(file_path)
# print(x.shape)
# x = df.clean_data(x)
# print(x.shape)


# class Data_base:

#     def get_data(self,file_path,nrows=1000):
#         df = pd.read_csv(file_path, sep=',', nrows=nrows)
#         print("data get : OK")
#         return df
    
#     def clean_data(self,df, test=False):
#         '''returns a DataFrame without outliers and missing values'''
#         print("data get : OK")
#         df = df.drop(["key"], axis=1)
#         df = df.dropna(how='any', axis='rows')
#         df = df[(df.fare_amount > 0) & (df.fare_amount < 70)]
#         df = df[(df.passenger_count < 8) & (df.passenger_count > 0)]
#         df = df[df["pickup_latitude"].between(left = 40, right = 42 )]
#         df = df[df["pickup_longitude"].between(left = -74.3, right = -72.9 )]
#         df = df[df["dropoff_latitude"].between(left = 40, right = 42 )]
#         df = df[df["dropoff_longitude"].between(left = -74, right = -72.9 )]
#         return df

