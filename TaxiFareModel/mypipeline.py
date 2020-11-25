from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


class MyPipeline():

    def __init__(self, trainer):
        self.trainer = trainer

    def create_pipeline(self, estimator):

        pipe_distance = make_pipeline(
            DistanceTransformer(),
            StandardScaler())

        cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

        feateng_blocks = [
            ('distance', pipe_distance, cols),
        ]

        features_encoder = ColumnTransformer(feateng_blocks)

        pipeline = Pipeline(steps=[
                    ('features', features_encoder),
                    ('model', estimator)])

        self.trainer.mlflow_log_metric("depuis le pipeline", 12345)

        return pipeline

