# imports
from TaxiFareModel.utils import compute_rmse
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """

        self.pipeline = None
        self.X = X
        self.y= y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)


    # def get_estimator(self):
    #     """Implement here"""
    #     estimator = self.kwargs.get("estimator", "RandomForest")
    #     if estimator == "RandomForest":
    #         model = RandomForestRegressor()
    #     elif estimator == "LinearRegression":
    #         model = LinearRegression()
    #     return model


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'),
                                OneHotEncoder())
        pipe_distance = make_pipeline(DistanceTransformer(),
                                      StandardScaler())

        # Define default feature engineering blocs
        distance_columns = [["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]]
        features_engineering_blocks = [
            ('distance', pipe_distance, distance_columns),
            ('time_features', pipe_time, ['pickup_datetime'])]


        features_encoder = ColumnTransformer(features_engineering_blocks,
                                             n_jobs=None,
                                             remainder="drop")

        regressor = RandomForestRegressor()

        self.pipeline = Pipeline(steps=[
                    ('features', features_encoder),
                    ('rgs', regressor)])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data(nrows=10_000)
    # clean data
    df = clean_data(df, test=False)
    # set X and y
    X= df.drop(columns='fare_amount')
    y = df['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    trainer = Trainer(X_train, y_train)
    # # train
    trainer.run()
    # # evaluate
    print(trainer.evaluate(X_test, y_test))




