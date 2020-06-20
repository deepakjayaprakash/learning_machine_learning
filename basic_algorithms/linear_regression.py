import numpy as np
from sklearn.linear_model import LinearRegression


def printStat(regr1):
    print(regr1.predict(np.array([[3, 5]])))
    print(regr1.score(X, y))
    # print(regr1.coef_)
    # print(regr1.intercept_)


if __name__ == '__main__':
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    print(X)

    # case: simple formula: y = a + b values of each element
    y = np.array([2, 3, 4, 5])
    regr = LinearRegression().fit(X, y)
    printStat(regr)

    # case: second element does not fit in the formula, so score wont be 1
    y = np.array([2, 3, 4, 5])
    regr = LinearRegression().fit(X, y)
    printStat(regr)

    y = np.dot(X, 3)  # result array is also a 2D
    regr = LinearRegression().fit(X, y)
    printStat(regr)
