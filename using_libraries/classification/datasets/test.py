"""
Performs some tests
"""

import clusters
import matplotlib.pyplot as plt

def plot(data):
    xdata = []
    ydata = []

    for d in data:
        xdata.append(d[0])
        ydata.append(d[1])

    ax = plt.gca()

    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)

    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    
    ax.scatter(xdata, ydata, c='blue')
    ax.legend(['Training Data', 'Predicted Data'])
    
    plt.show()


if (__name__ == "__main__"):
    dataset_1, x = clusters.gen_data([[10, 10], [25, 25]], 50, 1, 10, 0)
    dataset_1_var, x = clusters.gen_data([[10, 10], [25, 25]], 50, 1, 10, 0.3)

    plot(dataset_1)
    plot(dataset_1_var)

    dataset_2, x = clusters.gen_data([[10, 10], [-10, 10], [-10, -10], [10, -10]], 30, 5, 10, 0)
    dataset_2_var, x = clusters.gen_data([[10, 10], [-10, 10], [-10, -10], [10, -10]], 30, 5, 10, 0.3)

    plot(dataset_2)
    plot(dataset_2_var)

    dataset_3, x = clusters.gen_data([[3, 3], [-3, 3], [-3, -3], [3, -3]], 30, 5, 10, 0)
    dataset_3_var, x = clusters.gen_data([[3, 3], [-3, 3], [-3, -3], [3, -3]], 30, 5, 10, 0.3)

    plot(dataset_3)
    plot(dataset_3_var)

    dataset_4, x = clusters.gen_data([[1, 1], [2, 3], [-1, 2], [3, -2]], 30, 1, 3, 0)
    dataset_4_var, x = clusters.gen_data([[1, 1], [2, 3], [-1, 2], [3, -2]], 30, 1, 3, 0.3)

    plot(dataset_4)
    plot(dataset_4_var)
