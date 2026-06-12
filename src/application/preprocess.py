from abc import ABC, abstractmethod
import pandas as pd
import yaml
from src.application.models import Imputer


class BasePreprocess(ABC):
    @abstractmethod
    def transform(self):
        pass


class Preprocess(BasePreprocess):
    def __init__(
        self,
        imputer_config: str = "src/configs/imputer.yaml",
        column_config: str = "src/configs/columns.yaml",
        test: bool = False,
    ):
        self.imputer_config = imputer_config
        self.column_config = column_config
        with open(self.column_config, "r") as file:
            columns = yaml.safe_load(file)
            if test:
                self.columns = columns["test"]
            else:
                self.columns = columns["train"]
            self.imputercols = columns["imputer"]
        with open(self.imputer_config, "r") as file:
            self.imputer = Imputer.model_validate(yaml.safe_load(file)).model.build_model()
        self.columns = self.columns + ["x_mean", "y_mean", "z_mean", "TVT_mean"]

    def transform(self, dataset: pd.DataFrame):
        not_na = dataset[dataset["GR"].notna()]
        na = dataset[dataset["GR"].isna()]
        self.imputer.fit(not_na[self.imputercols], not_na["GR"])
        na["GR"] = self.imputer.predict(na[self.imputercols])
        data = pd.concat([not_na, na], axis=0)
        train = data[data["TVT_input"].notna()][self.imputercols + ["TVT_input"]]
        evaluation = data[data["TVT_input"].isna()]
        last_vals = min(train.shape[0], 20)
        train = train.tail(last_vals)
        train = pd.DataFrame(
            {
                "x_mean": [train["X"].mean()],
                "y_mean": [train["Y"].mean()],
                "z_mean": [train["Z"].mean()],
                "TVT_mean": [train["TVT_input"].mean()],
            }
        )
        data = pd.merge(data, train, how="cross")
        return data
