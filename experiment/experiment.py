import numpy as np
import matplotlib
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    x = np.random.rand(40)
    y = 2 * x - 1 + np.random.rand(40)
    print("x", x)
    print("y", y)
    plt.scatter(x,y)
    plt.show()