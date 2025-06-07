"""
Trigonometric dataset for some ML.

The general idea is that we are going to have definitions for all integer values.
This gives us the opportunity to have regression models that extend the data beyond 
the generated bounds or to have regresiion models that fill in the blanks between
the dots.

The function implemented is:
    Y = asin(X) + bcos(X)
"""

import numpy as np
from math import sin, cos
from random import randint, uniform

def gen_data(lower_bound: int = 0, upper_bound: int = 10, variance: float = 0) -> dict:
    """
    Generates the data and returns it all in a dictionary.

    Args:
        lower_bound: starting point for X
        upper_bound: end point for X
        variance: maximum percentage of error that each point experiences
    """

    data = {"X": np.array([np.array([i]) for i in np.arange(lower_bound, upper_bound+1, 1)]), "Y": np.empty((upper_bound - lower_bound) + 1)}

    a = randint(-5, 5)
    b = randint(-5, 5)

    print(f"{a}sin(X) + {b}cos(X)")

    for i in range(lower_bound, upper_bound+1, 1):
        perror = 1 + uniform(-variance, variance)
        data["Y"][i] = (a*sin(i) + b*cos(i)) * perror

    return data
