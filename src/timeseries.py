import sys
import os
import numpy as np
import pandas as pd
from loguru import logger


class Timeseries:
    def __init__(
        self,
        data_files: str,
        input_params: str,
        output_params: str,
        data_dir: str = ".",
        norm_type: str = "none",
        time_lag: int = 0,
        future_time: int = 0,
    ):
        self.data_dir = data_dir
        self.input_names = [x.strip() for x in input_params.split(" ")]
        self.output_names = [x.strip() for x in output_params.split(" ")]
        self.file_names = [x.strip() for x in data_files.split(" ")]
        self.load_data_from_files(self.input_names, self.output_names)
        input_padding = np.zeros((time_lag, len(self.input_names)), dtype=np.float32)
        norm_fun = self.normalization(norm_type)
        self.input_data = np.array(norm_fun(self.input_data))
        if future_time != 0:
            self.input_data = self.input_data[:-future_time]
        train_test_mark = int(len(self.input_data) * 0.8)
        self.train_input = self.input_data[:train_test_mark]
        self.test_input = self.input_data[train_test_mark:]
        self.train_input = np.append(input_padding, self.train_input, axis=0)
        self.test_input = np.append(input_padding, self.test_input, axis=0)
        self.output_data = np.array(norm_fun(self.output_data))[future_time:]
        self.train_output = self.output_data[:train_test_mark]
        self.test_output = self.output_data[train_test_mark:]

    def normalization(self, norm_type: str = "minmax"):
        normalization_types = {
            "minmax": self.min_max_normalize,
            "mean_std": self.mean_normalize,
            "none": self.none_normalize,
        }
        logger.info(f"using {norm_type.upper()} Normalization")
        return normalization_types[norm_type]

    def load_data_from_files(self, input_names, output_names):
        logger.info(f"loading data from {self.file_names[0]}")
        self.input_data, self.output_data = self.load_data(
            "/".join([self.data_dir, self.file_names[0]]),
            input_names,
            output_names,
        )
        if len(self.file_names) > 1:
            for name in self.file_names[1:]:
                logger.info(f"loading data from {name}")
                input_data, output_data = self.load_data(
                    "/".join([self.data_dir, name]), self.input_names, self.output_names
                )
                self.input_data = self.input_data.append(input_data)
                self.output_data = self.output_data.append(output_data)
        # return self.input_data, self.output_data

    def load_data(self, file_name, in_params, out_params) -> None:
        if not os.path.exists(file_name):
            logger.error(f" File: {file_name} Does Not Exist")
            sys.exit()
        all_param_names = list(set(self.input_names + self.output_names))
        data = pd.read_csv(file_name, sep=",", skipinitialspace=True)
        data = data[all_param_names].astype(np.float32)
        return data[in_params], data[out_params]

    def none_normalize(self, data):
        return data

    def mean_normalize(self, data):
        return (data - data.mean()) / data.std()

    def min_max_normalize(self, data):
        for x in data:
            min_ = data[x].min()
            max_ = data[x].max()
            if min_ == max_:
                min_ = 0.0
            data[x] = (data[x] - min_) / (max_ - min_)
        return data
