"""
Tests all the datasets
"""

import linear
import poly
import trig

from types import ModuleType
import matplotlib.pyplot as plt


def plot(data: dict) -> None:
    """
    Handles the plotting.
    """

    plt.plot(data['X'], data['Y'], marker='.', linestyle='None')
    plt.show()


def test_data(module: ModuleType) -> None:
    """
    Create a plot with some different bounds for a particular dataset.
    Neat that I get to use the ModuleType thing. 
    """

    data1 = module.gen_data()
    data2 = module.gen_data(0, 20)
    data3 = module.gen_data(-10, 10)
    data4 = module.gen_data(0, 10)

    plot(data1)
    plot(data2)
    plot(data3)
    plot(data4)


if (__name__ == "__main__"):
    test_data(linear)
    test_data(poly)
    test_data(trig)
