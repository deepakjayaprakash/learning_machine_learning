from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def print_correct_wrong(y_pred, y_test):
    count_wrong = 0
    count_correct = 0
    for i in y_test:
        # print(y_test[i], ",", y_pred[i])
        if y_test[i] == y_pred[i]:
            count_correct = count_correct + 1
        else:
            count_wrong = count_wrong + 1
    print("Correct count : ", count_correct)
    print("Wrong count : ", count_wrong)


def print_data_split(X_test, X_train):
    print("testing shape : ", X_train.shape)  # prints tuple of (rows, columns)
    print("training data shape : ", X_test.shape)


def print_data(X, feature_names, target_names, Y):
    print("Feature names:", feature_names)
    print("Target names:", target_names)
    # print("\nFirst 10 rows of X:\n", X[:10])
    # print("\nFirst 10 rows of Y:\n", Y[:10])


if __name__ == '__main__':
    print("started iris test")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    print_data(X, feature_names, target_names, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    print_data_split(X_test, X_train)
    model = linear_model.LogisticRegression().fit(X_train, y_train)
    print(model.score(X_train, y_train))


    y_pred = model.predict(X_test)
    print("Predicted : ", y_pred)
    print("Actual : ", y_test)
    print_correct_wrong(y_pred, y_test)