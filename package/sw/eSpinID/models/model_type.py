# models/model_type.py
from enum import Enum

from eSpinID.models.xgboost import XGBoostSurrogate


class ModelTypeError(Exception):
    pass


class ModelType(Enum):
    XGBoost = "XGBoostSurrogate"

    @staticmethod
    def from_str(label):
        if label == 'XGBoostSurrogate':
            return ModelType.XGBoost
        else:
            raise ModelTypeError('Unknown model type: ' + label)

    def create(self, config):
        if self is ModelType.XGBoost:
            return XGBoostSurrogate(config)
        else:
            raise ModelTypeError('Unknown model type: ' + str(self))