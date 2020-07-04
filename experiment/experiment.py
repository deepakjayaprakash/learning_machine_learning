import datetime

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    # frame = pd.DataFrame(columns= ['A','B', 'value'])
    frame = pd.DataFrame(columns=['A', 'value'])
    daet1 = datetime.datetime.strptime('2020-06-10', "%Y-%m-%d").toordinal()
    daet2 = datetime.datetime.strptime('2020-06-11', "%Y-%m-%d").toordinal()
    frame.loc[0] = [daet1, 234]
    frame.loc[1] = [daet2, 235]

    # frame.loc[0] = ['A', 234]
    # frame.loc[1] = ['B', 235]
    print(frame.head())
    feature_train, feature_test, result_train, result_test = train_test_split(frame[['A']], frame['value'],
                                                                              test_size=0.5)
    print(type(feature_train))
    regr = linear_model.LinearRegression()

    regr.fit(feature_train, result_train)
    print("regression_done")
    # print("score: ", regr.score(feature_train, result_train))
    result_pred = regr.predict(feature_test)
    print("pred:", result_pred)
