import numpy as np

def haversine_vectorized(df, start_lat="pickup_latitude", start_lon="pickup_longitude",
                         end_lat="dropoff_latitude", end_lon="dropoff_longitude"):

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)), np.radians(df[start_lon].astype(float))
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)), np.radians(df[end_lon].astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def compute_rmse(y_pred, y_true):
    # y_pred = np.asarray(y_pred)
    # y_true = np.asarray(y_true)
    return np.sqrt(np.mean((y_true - y_pred)**2))