import mlflow
from src.application.models import Logger, Estimator


class ModelLogger:
    def __init__(self, experiment, tracking_uri: str = "http://127.0.0.1:5000"):
        self.experiment = experiment
        self.logger = Logger("TRAINING", "logs/training.log").logger
        self.tracking_uri = tracking_uri

    def log_model(self, model: Estimator):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)
            self.logger.info(f"Experiment Name: {self.experiment}")
            with mlflow.start_run() as run:
                mlflow.set_tag("Model", model.name)
                mlflow.log_params(model.get_params())
                model_info = mlflow.xgboost.log_model(model.model, artifact_path="xgboost")
                print(model_info.model_uri)
                # mlflow.register_model(model_info.model_uri,name="xgboost")
        except Exception as e:
            self.logger.error(f"Experiment Failed:{e}")
