from gpbasics import global_parameters as global_param

global_param.ensure_init()

from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import logging


def get_default_path() -> str:
    return "data/"


class DatasetHandler:
    """
    DatasetHandler manages reading a dataset from file source and provides a general interface for all different kinds
    of datasets.
    """

    def __init__(self):
        self.x: np.ndarray = None
        self.y: np.ndarray = None
        self.sample: int = None
        self.max_y: float
        self.min_y: float
        self.max_x: List[float]
        self.min_x: List[float]

    def get_bare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def init_data(self):
        x, y = self.get_bare_data()

        self.init_by_predefined_data(x, y)

    def init_by_predefined_data(self, x: np.ndarray, y: np.ndarray):
        if global_param.p_scale_data_y:
            self.min_y = min(y)
            y = y - self.min_y
            self.max_y = max(y)
            y = y / self.max_y
        else:
            self.min_y = min(y)
            self.max_y = max(y)

        self.y = y.reshape(-1, 1)
        self.max_x = []
        self.min_x = []

        if len(x.shape) == 2 and x.shape[1] > 1:
            for i in range(x.shape[1]):
                min_x = min(x[:, i])
                x[:, i] = x[:, i] - min_x
                max_x = max(x[:, i])
                x[:, i] = x[:, i] / max_x
                self.min_x.append(min_x)
                self.max_x.append(max_x)
            self.x = x.reshape(-1, x.shape[1])
        else:
            min_x = min(x)
            x = x - min_x
            max_x = max(x)
            x = x / max_x
            self.min_x.append(min_x)
            self.max_x.append(max_x)
            self.x = x.reshape(-1, 1)

    def get_x_range(self) -> Tuple[float, float]:
        return float(min(self.x)), float(max(self.x))

    def get_y_range(self) -> Tuple[float, float]:
        return float(min(self.y)), float(max(self.y))


class GeneralDatasetHandler(DatasetHandler):
    def __init__(self, filename: str, y_col_name: str, x_col_name: str = None,
                 name: str = "A dataset", sample: int = None):
        """
        :param filename: name of the CSV-file of the dataset. dataset has to be stored in the relative path "/data"
        :param y_col_name: name of the column storing the target variable
        :param x_col_name: optional_parameter, name of the column storing the input variable. If x_col_name is not provided, a simple index is assumed.
        :param name: optional_parameter, short and unambiguous dataset name used for further referencing throughout the code
        :param sample: optional parameter, if given the dataset is cut to the given sample size. No dataset amplification!
        """

        assert not isinstance(y_col_name, list), "target value y must be uni-dimensional"

        super(GeneralDatasetHandler, self).__init__()
        self.name: str = name
        self.filename: str = filename

        self.filename: str = get_default_path() + filename

        self.x_col_name: str = x_col_name
        self.y_col_name: str = y_col_name
        self.sample: int = sample
        self.init_data()

    def get_bare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.sample is not None:
            gc = pd.read_csv(self.filename, error_bad_lines=False, nrows=self.sample).dropna()
        else:
            gc = pd.read_csv(self.filename, error_bad_lines=False).dropna()

        gc = gc._get_numeric_data()

        if self.x_col_name == "§ALL":
            logging.info("All columns of dataset are used as input data except for target value.")
            x_col_name = [c for c in gc.columns if c != self.y_col_name]

            self.x_col_name = []

            for col in x_col_name:
                col_data = np.array(gc[col])
                if np.max(col_data) != np.min(col_data):
                    self.x_col_name.append(col)

        if self.x_col_name is not None:
            if type(self.x_col_name) is list:
                gc.sort_values(by=self.x_col_name)
            else:
                gc.sort_values(by=[self.x_col_name])

        y = np.array(gc[self.y_col_name])

        if self.x_col_name is None:
            x = np.linspace(0, len(y), len(y), endpoint=False)
        else:
            x = np.array(gc[self.x_col_name])

        return x, y


class SolarIrradianceHandler(GeneralDatasetHandler):
    def __init__(self):
        super(SolarIrradianceHandler, self).__init__("d1_solar_irradiance.csv", "cycle", "year", name="SolarIr")
        self.init_data()


class MaunaLoaHandler(GeneralDatasetHandler):
    def __init__(self):
        super(MaunaLoaHandler, self).__init__("d2_maunaloa.csv", 'Carbon Dioxide (ppm)', 'Decimal Date',
                                              name="MaunaLoa")


class PowerPlantHandler(GeneralDatasetHandler):
    def __init__(self):
        super(PowerPlantHandler, self).__init__("d3_powerplant.csv", y_col_name="EP",
                                                x_col_name=["AT", "V", "AP", "RH"], name="PowerPlant")
        self.init_data()


class GEFCOMHandler(GeneralDatasetHandler):
    def __init__(self):
        super(GEFCOMHandler, self).__init__("d4_gefcom.csv", 'load', 'timestamp', name="GEFCOM")
        self.init_data()


class TemperatureHandler(GeneralDatasetHandler):
    def __init__(self):
        super(TemperatureHandler, self).__init__("d8_temperature.csv", "y", "X", name="Temp")
        self.init_data()


class BirthsHandler(GeneralDatasetHandler):
    def __init__(self):
        super(BirthsHandler, self).__init__("d15_births.csv", "y", "X", name="Births")
        self.init_data()
