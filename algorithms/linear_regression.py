import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def printStat(regr1):
    print(regr1.predict(np.array([[3, 5]])))
    print(regr1.score(X, y))
    # print(regr1.coef_)
    # print(regr1.intercept_)


def regression_on_diabetes_data_set_array():
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)
    print(regr.score(diabetes_X_train, diabetes_y_train))
    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.scatter(diabetes_X_test, diabetes_y_pred, color='red')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    plt.show()


def basic_regression():
    global X, y
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


def regression_on_diabetes_data_set_data_frame():
    diabetes_data = datasets.load_diabetes(as_frame=True)
    feature_names = diabetes_data.feature_names
    print("feature_names: ", feature_names)

    X = diabetes_data.data
    y = diabetes_data.target
    diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    print("train size", diabetes_X_train.shape)
    print("test size", diabetes_X_test.shape)

    regr = linear_model.LinearRegression()

    regr.fit(diabetes_X_train, diabetes_y_train)
    print(regr.score(diabetes_X_train, diabetes_y_train))
    diabetes_y_pred = regr.predict(diabetes_X_test)
    print("expected_results", diabetes_y_test)
    print("actual_results", diabetes_y_pred)
    print("mae", mean_absolute_error(diabetes_y_test, diabetes_y_pred))


if __name__ == '__main__':
    # basic_regression()

    # regression_on_diabetes_data_set_array()
    regression_on_diabetes_data_set_data_frame()
