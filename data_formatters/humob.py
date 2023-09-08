# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for humob dataset.

Defines dataset specific column definitions and data transformations. Uses
entity specific z-score normalization.
"""

import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class HumobFormatter(GenericDataFormatter):
    """Defines and formats data for the humob dataset.

    Note that per-entity z-score normalization is used here, and is implemented
    across functions.

    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.

            (
            "x",
            DataTypes.REAL_VALUED,
            InputTypes.TARGET,
        ),  # 待定，这里是否应该(x,y)合为一个categorical变量
        (
            "y",
            DataTypes.REAL_VALUED,
            InputTypes.TARGET,
        ),  # 待定，这里是否应该(x,y)合为一个categorical变量
    """

    _column_definition = [
        ("uid", DataTypes.REAL_VALUED, InputTypes.ID),  # 用户唯一ID
        ("timestamp", DataTypes.DATE, InputTypes.TIME),  # 时间的指标
        # ("poi", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),  # 待定为poi相关变量
        (
            "d",
            DataTypes.REAL_VALUED,
            InputTypes.KNOWN_INPUT,
        ),  # 待定，tft貌似能吸收很多input变量
        (
            "location_id",
            DataTypes.CATEGORICAL,
            InputTypes.TARGET,
        ),  # 地点ID
        (
            "categorical_id",
            DataTypes.CATEGORICAL,
            InputTypes.STATIC_INPUT,
        ),  # 用户id作为静态输入
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self._time_steps = self.get_fixed_params()["total_time_steps"]

    def split_data(self, df, valid_boundary=53, test_boundary=60):
        """Splits data frame into training-validation-test data frames.
        按照时间维度划分而非用户维度, 相关资料见notion, 需讨论
        0-74天

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_boundary: Starting day for validation data
          test_boundary: Starting day for test data

        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print("Formatting train-valid-test splits.")

        index = df["d"]  # 因此需要保留原有的d column，这里需要依次做划分
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
        test = df.loc[(index >= test_boundary) & (index < 75)]

        self.set_scalers(df)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.
        归一化

        Args:
          df: Data to use to calibrate scalers.
        """
        print("Setting scalers with training data...")

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(
            InputTypes.ID, column_definitions
        )
        # target_column = utils.get_single_col_by_input_type(
        # InputTypes.TARGET, column_definitions
        # )

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        # Initialise scaler caches
        self._real_scalers = {}
        # self._target_scaler = {}
        identifiers = []
        for identifier, sliced in df.groupby(id_column):
            if len(sliced) >= self._time_steps:
                data = sliced[real_inputs].values
                # targets = sliced[[target_column]].values
                self._real_scalers[
                    identifier
                ] = sklearn.preprocessing.StandardScaler().fit(data)

                # self._target_scaler[
                # identifier
                # ] = sklearn.preprocessing.StandardScaler().fit(targets)
            identifiers.append(identifier)

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                srs.values
            )
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._target_scaler = categorical_scalers
        self._num_classes_per_cat_input = num_classes

        # Extract identifiers in case required
        self.identifiers = identifiers

    def transform_inputs(self, df):
        """Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.

        Args:
          df: Data frame to transform.

        Returns:
          Transformed data frame.

        """

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError("Scalers have not been set!")

        # Extract relevant columns
        column_definitions = self.get_column_definition()
        id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        # Transform real inputs per entity
        df_list = []
        for identifier, sliced in df.groupby(id_col):
            # Filter out any trajectories that are too short
            print(f"Length of original sliced: {len(sliced)}")
            if len(sliced) >= self._time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
                    sliced_copy[real_inputs].values
                )
                df_list.append(sliced_copy)
                print(f"Length of current {identifier} in df_list: {len(sliced_copy)}")

        output = pd.concat(df_list, axis=0)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
          predictions: Dataframe of model predictions.

        Returns:
          Data frame of unnormalised predictions.
        """

        if self._target_scaler is None:
            raise ValueError("Scalers have not been set!")

        column_names = predictions.columns

        df_list = []
        for identifier, sliced in predictions.groupby("identifier"):
            sliced_copy = sliced.copy()
            target_scaler = self._target_scaler[identifier]

            for col in column_names:
                if col not in {"forecast_time", "identifier"}:
                    reshaped_data = sliced_copy[col].values.reshape(-1, 1)
                    sliced_copy[col] = target_scaler.inverse_transform(reshaped_data)
            df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            "total_time_steps": 7 * 48,  # 作为有效数据的最低时间点数量，对三个数据集同样
            # 需要学习的过去的时间点数 + 需要预测的时间点数
            # 需要补全”不动“的时间戳
            "num_encoder_steps": 6 * 48,
            "num_epochs": 100,
            "early_stopping_patience": 5,
            "multiprocessing_workers": 5,
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            "dropout_rate": 0.1,
            "hidden_layer_size": 160,
            "learning_rate": 0.001,
            "minibatch_size": 20,  # 在 Transformer 模型中，批量大小是指在神经网络的单次向前和向后传递过程中并行处理
            # 的输入示例（序列）的数量。 它是训练模型时可以调整的超参数，是训练效率和性能的关键因素。
            "max_gradient_norm": 0.01,
            "num_heads": 4,
            "stack_size": 1,
        }

        return model_params

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.

        Use to sub-sample the data for network calibration and a value of -1 uses
        all available samples.

        Returns:
          Tuple of (training samples, validation samples)
        """
        return 450000, 50000
