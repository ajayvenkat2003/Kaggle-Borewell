from sklearn.neighbors import KNeighborsRegressor
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Union, Literal, Optional, Any
from xgboost import XGBRegressor
import logging


class BaseImputer(ABC, BaseModel):
    KIND: str

    @abstractmethod
    def build_model(self):
        pass


class KNeighborsImputer(BaseImputer):
    KIND: Literal["KNN"]
    n_neighbors: int = 3

    def build_model(self):
        return KNeighborsRegressor(n_neighbors=self.n_neighbors)


class Imputer(BaseModel):
    model: Union[KNeighborsImputer] = Field(..., discriminator="KIND")


class BaseEstimator(ABC, BaseModel):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def get_params(self):
        pass


class XGBModel(BaseEstimator):
    name: Literal["xgboost"]
    n_estimators: int = 500
    max_depth: int = 2
    learning_rate: float = 0.05
    metrics: Optional[dict] = None
    model: Optional[Any] = None

    def build_model(self):
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
        )
        return self.model

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }


class Estimator(BaseModel):
    model: Union[XGBModel] = Field(..., discriminator="name")


class Logger:
    def __init__(self, name, handler):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(name)
        file_handler = logging.FileHandler(handler)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)
