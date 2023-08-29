import pandas as pd

from typing import Tuple, Union, List

import xgboost as xgb

import pickle

# load model from pickle file
model = pickle.load(open("model.pk1", 'rb'))


class DelayModel:

    def __init__(
        self
    ):
        self._model = model # Model should be saved in this attribute.

    def preprocess(
        data: pd.DataFrame,
        target_column: str = 'delay'): 
    #-> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        data = pd.DataFrame(list(data.values())[0][0], index=[0])

        data['delay'] = 1

        features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1)
        
        target = data[target_column]

        return features, target

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        return model.fit(features, target)

    def predict(
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        return model.predict(features)