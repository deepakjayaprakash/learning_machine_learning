from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# The entire code is just to learn, you will find this code on tutorials point under the topic scikit_learn
from sklearn.tree import DecisionTreeClassifier


def get_correct_wrong(count_correct, count_wrong, y_pred, y_test):
    for i in y_test:
        # print(y_test[i], ",", y_pred[i])
        if y_test[i] == y_pred[i]:
            count_correct = count_correct + 1
        else:
            count_wrong = count_wrong + 1
    return count_correct, count_wrong


def print_metrics(y_pred, y_test):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    count_correct = 0
    count_wrong = 0
    count_correct, count_wrong = get_correct_wrong(count_correct, count_wrong, y_pred, y_test)
    print("count_correct : ", count_correct, "count_wrong : ", count_wrong)


def decision_tree(X_test, X_train, y_test, y_train):
    print("================================================")
    print("Decision Tree")
    dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
    dtree_predictions = dtree_model.predict(X_test)

    # creating a confusion matrix
    cm = confusion_matrix(y_test, dtree_predictions)
    print_metrics(dtree_predictions, y_test)
    print(cm)


def naive_bayes(X_test, X_train, y_test, y_train):
    print("================================================")
    print("Decision Tree")
    gnb = GaussianNB().fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)

    # creating a confusion matrix
    cm = confusion_matrix(y_test, gnb_predictions)
    print_metrics(gnb_predictions, y_test)
    print(cm)


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
        X, y, test_size=0.2, random_state=1
    )
    print_data_split(X_test, X_train)
    # classify_knn(X_test, X_train, y_test, y_train, iris)
    # decision_tree(X_test, X_train, y_test, y_train)
    naive_bayes(X_test, X_train, y_test, y_train)


def knn(X_test, X_train, y_test, y_train, iris):
    print("================================================")
    print("KNN ")
    classifier_knn = KNeighborsClassifier(n_neighbors=3)
    classifier_knn.fit(X_train, y_train)
    y_pred = classifier_knn.predict(X_test)
    # Finding accuracy by comparing actual response values(y_test)with predicted response value(y_pred)
    print_metrics(y_pred, y_test)

    sample = [[5, 5, 3, 2], [2, 4, 3, 5]]
    preds = classifier_knn.predict(sample)
    pred_species = [iris.target_names[p] for p in preds]
    print("Predictions:", pred_species)

    # Model Persistance
    # from sklearn.externals import joblib
    # joblib.dump(classifier_knn, 'iris_classifier_knn.joblib')
    # joblib.load('iris_classifier_knn.joblib')
