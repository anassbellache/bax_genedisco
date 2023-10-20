import datetime
import time
from argparse import Namespace
from typing import List, Dict, AnyStr, Tuple, Union, Optional

import numpy as np
from slingpy import AbstractDataSource


class ListDataSource(AbstractDataSource):
    def __init__(self, data, included_indices: List[int]):
        super().__init__(included_indices)
        self.data_list = list(data)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> List[np.ndarray]:
        return self.data_list[index]

    def get_shape(self) -> List[Tuple[Union[Optional[int]]]]:
        return self.data_list.shape()

    def _get_data(self) -> List[np.ndarray]:
        return self.data_list

    def get_by_row_name(self, row_name: AnyStr) -> List[np.ndarray]:
        pass

    def get_row_names(self) -> List[AnyStr]:
        pass

    def _get_all_row_names(self) -> List[AnyStr]:
        pass

    def get_column_names(self) -> List[List[AnyStr]]:
        pass

    def get_column_code_lists(self) -> List[List[Dict[int, AnyStr]]]:
        pass


def dict_to_namespace(params):
    # If params is a dict, convert to Namespace
    if isinstance(params, dict):
        params = Namespace(**params)

    return params


def jaccard_similarity(list1, list2):
    """Return jaccard similarity between two sets."""
    s1 = set(list1)
    s2 = set(list2)
    jac_sim = float(len(s1.intersection(s2)) / len(s1.union(s2)))
    return jac_sim


class Timer(object):
    """
    Timer class. Thanks to Eli Bendersky, Josiah Yoder, Jonas Adler, Can Kavaklıoğlu,
    and others from https://stackoverflow.com/a/50957722.
    """

    def __init__(self, name=None, filename=None):
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.2f seconds' % (time.time() - self.tstart)
        if self.name:
            message = '*[TIME] [%s] ' % self.name + message
        print(message)
        if self.filename:
            with open(self.filename, 'a') as file:
                print(str(datetime.datetime.now()) + ": ", message, file=file)
