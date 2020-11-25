# # imports
# from TaxiFareModel.utils import compute_rmse
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.ensemble import RandomForestRegressor
# from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import LinearRegression
# from TaxiFareModel.data import get_data, clean_data
# from sklearn.model_selection import train_test_split
# from TaxiFareModel.utils import haversine_vectorized
# import numpy as np

# class Trainer:
#     def __init__(self, X, y):
#         """
#             X: pandas DataFrame
#             y: pandas Series
#         """

#         self.pipeline = None
#         self.X = X
#         self.y= y
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)


#     # def get_estimator(self):
#     #     """Implement here"""
#     #     estimator = self.kwargs.get("estimator", "RandomForest")
#     #     if estimator == "RandomForest":
#     #         model = RandomForestRegressor()
#     #     elif estimator == "LinearRegression":
#     #         model = LinearRegression()
#     #     return model


#     def set_pipeline(self):
#         """defines the pipeline as a class attribute"""
#         pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'),
#                                 OneHotEncoder())
#         pipe_distance = make_pipeline(DistanceTransformer(),
#                                       StandardScaler())

#         # Define default feature engineering blocs
#         distance_columns = ["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]
#         features_engineering_blocks = [
#             ('distance', pipe_distance, distance_columns),
#             ('time_features', pipe_time, ['pickup_datetime'])]


#         features_encoder = ColumnTransformer(features_engineering_blocks,
#                                              n_jobs=None,
#                                              remainder="drop")

#         regressor = RandomForestRegressor()

#         self.pipeline = Pipeline(steps=[
#                     ('features', features_encoder),
#                     ('rgs', regressor)])

#     def run(self):
#         """set and train the pipeline"""
#         self.set_pipeline()
#         self.pipeline.fit(self.X_train, self.y_train)

#     def evaluate(self, X_test, y_test):
#         """evaluates the pipeline on df_test and return the RMSE"""
#         y_pred = self.pipeline.predict(self.X_test)
#         rmse = compute_rmse(y_pred, self.y_test)
#         return rmse


# if __name__ == "__main__":
#     # get data
#     df = get_data(nrows=10_000)
#     # clean data
#     df = clean_data(df, test=False)
#     # set X and y
#     X= df.drop(columns='fare_amount')
#     y = df['fare_amount']
#     # hold out
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#     trainer = Trainer(X_train, y_train)
#     # # train
#     trainer.run()
#     # # evaluate
#     print(trainer.evaluate(X_test, y_test))

# imports
# from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
# from TaxiFareModel.data import get_data,clean_data
# from sklearn.model_selection import train_test_split
# from  sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import pandas as pd
# import numpy as np
# from memoized_property import memoized_property
# import mlflow
# from mlflow.tracking.client import MlflowClient

# MLFLOW_URI = "https://mlflow.lewagon.co/"
# myname = "ellyn"
# EXPERIMENT_NAME = f"TaxifareModel_{myname}"

# class Trainer():
#     def __init__(self, X, y, experiment_name):
#         """
#             X: pandas DataFrame
#             y: pandas Series
#         """
#         self.pipeline = None
#         self.X = X
#         self.y = y
#         self.experiment_name = experiment_name

#     def set_pipeline(self):
#         """defines the pipeline as a class attribute"""
#         pipe_distance = make_pipeline(DistanceTransformer(),StandardScaler())
#         pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder(handle_unknown='ignore'))
#         dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
#         time_cols = ['pickup_datetime']
#         feat_eng_bloc = ColumnTransformer([('time', pipe_time, time_cols),
#                                         ('distance', pipe_distance, dist_cols)])
#         self.pipeline = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc),
#                             ('regressor',RandomForestRegressor())])
#         return self.pipeline

#     def run(self):
#         """set and train the pipeline"""
#         self.mlflow_run()
#         self.pipeline=self.set_pipeline()
#         self.pipeline.fit(self.X, self.y)
#         self.mlflow_log_param('estimator', 'random_forest')

#     def evaluate(self, X_test, y_test):
#         """evaluates the pipeline on df_test and return the RMSE"""
#         y_pred=self.pipeline.predict(X_test)
#         rmse = np.sqrt(((y_pred - y_test)**2).mean())
#         self.mlflow_log_metric('rmse', rmse)


#     @memoized_property
#     def mlflow_client(self):
#         mlflow.set_tracking_uri(MLFLOW_URI)
#         return MlflowClient()

#     @memoized_property
#     def mlflow_experiment_id(self):
#         try:
#             return self.mlflow_client.create_experiment(self.experiment_name)
#         except BaseException:
#             return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

#     @memoized_property
#     def mlflow_run(self):
#         return self.mlflow_client.create_run(self.mlflow_experiment_id)

#     def mlflow_log_param(self, key, value):
#         self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

#     def mlflow_log_metric(self, key, value):
#         self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


# if __name__ == "__main__":
#     # get data and clean
#     df=clean_data(get_data())
#     # set X and y
#     X=df.drop('fare_amount',axis=1)
#     y=df.fare_amount
#     # hold out
#     X_train,X_test,y_train,y_test=train_test_split(X,y)
#     trainer=Trainer(X_train,y_train, EXPERIMENT_NAME)
#     # train
#     trainer.run()
#     # params = trainer.run().get_params()

#     # evaluate
#     print(trainer.evaluate(X_test,y_test))
#     # print(params)



from TaxiFareModel.data import get_data, clean_df, holdout
from TaxiFareModel.pipeline import MyPipeline
from TaxiFareModel.metrics import compute_rmse
from TaxiFareModel.mlflowbase import MLFlowBase

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class Trainer(MLFlowBase):

    def __init__(self):

        super().__init__("ellyn",
                         "https://mlflow.lewagon.co")

    def evaluate(self):

        y_pred = self.model.predict(self.X_val)

        # process rmse
        rmse = compute_rmse(y_pred, self.y_val)

        # mlflow metrics
        self.mlflow_log_metric("rmse", rmse)

    def create_estimator(self, model_name):

        if model_name == 'linear':
            model = LinearRegression()
        elif model_name == 'rf':
            model = RandomForestRegressor()

        return model

    def fit_model(self, model_name, hyperparams):

        grid_search = GridSearchCV(
            self.model,
            param_grid=hyperparams[model_name],
            cv=5
        )

        grid_search.fit(self.X_train, self.y_train)
        score = grid_search.score(self.X_val, self.y_val)

        # mlflow metrics
        self.mlflow_log_metric("xval_score", score)

        self.model = grid_search.best_estimator_

        for k, v in grid_search.best_params_.items():
            self.mlflow_log_param(k, v)


    def train(self, model_name, hyperparams):

        # create mlflow run
        self.mlflow_create_run()

        # get data
        df = get_data()

        # clean data
        df = clean_df(df)

        # mlflow param
        self.mlflow_log_param("estimator", model_name)

        # create pipeline
        self.pipeline = TotoPipeline(self)

        dyn_model = self.create_estimator(model_name)

        self.model = self.pipeline.create_pipeline(dyn_model)

        # get df
        self.X_train, self.X_val, self.y_train, self.y_val = holdout(df)

        self.fit_model(model_name, hyperparams)

        self.evaluate()

        return self


if __name__ == '__main__':

    models = ['linear', 'rf']
    hyperparams = {
        'linear': {
                'features__distance__standardscaler__with_mean': [False, True],
        },
        'rf': {
                'features__distance__standardscaler__with_mean': [False, True],
                'model__min_weight_fraction_leaf': [0.0, 0.1]
        }
    }

    for model in models:

        trainer = Trainer()
        trainer.train(model, hyperparams)


