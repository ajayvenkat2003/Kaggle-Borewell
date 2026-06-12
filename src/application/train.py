import yaml
from src.io.dataset import Files, Dataset
from src.application.preprocess import Preprocess
from src.application.models import Estimator, Logger
from src.application.modelLogger import ModelLogger


class Trainer:
    def __init__(self, preprocessor, config="src/configs/train.yaml"):
        self.preprocessor = preprocessor
        with open(config, "r") as file:
            self.config = yaml.safe_load(file)
            self.location = self.config["location"]
            self.target = self.config["target_column"]
            self.estimator = Estimator.model_validate(self.config["estimator"]).model
            self.logger = Logger("TRAINING", "logs/training.log").logger
            self.modelLogger = ModelLogger(experiment="experiment 1")

    def train(self):
        files = Files(self.location)
        dataset = Dataset(files)
        self.logger.info("preprocessing started")
        data = dataset.processData(self.preprocessor)
        self.logger.info("preprocessing completed")
        data[self.target] = data[self.target] - data["TVT_mean"]
        target = data.pop(self.target)
        self.logger.info(f"training size = {target.shape[0]}")
        self.logger.info(f"Model:{self.estimator.name}")
        self.model = self.estimator.build_model()
        self.logger.info("training started")
        self.model.fit(data, target)
        self.logger.info("training completed")
        self.modelLogger.log_model(self.estimator)


def main():
    preprocessor = Preprocess(test=False)
    trainer = Trainer(preprocessor=preprocessor)
    trainer.train()


if __name__ == "__main__":
    main()
