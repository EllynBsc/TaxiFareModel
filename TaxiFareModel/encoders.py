from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized
import numpy as np
import pandas as pd

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        timezone_name = 'America/New_York'
        time_column = "pickup_datetime"
        X_new = X.copy()
        X_new.index = pd.to_datetime(X_new[time_column])
        X_new.index = X_new.index.tz_convert(timezone_name)
        X_new["dow"] = X_new.index.weekday
        X_new["hour"] = X_new.index.hour
        X_new["month"] = X_new.index.month
        X_new["year"] = X_new.index.year
        X_new.reset_index(drop=True)
        return X_new[['dow', 'hour', 'month', 'year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        X['distance'] = haversine_vectorized(X,
         start_lat="pickup_latitude",
         start_lon="pickup_longitude",
         end_lat="dropoff_latitude",
         end_lon="dropoff_longitude")

        return X[['distance']]
