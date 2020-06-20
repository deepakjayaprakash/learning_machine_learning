import math

import pandas as pd
import quandl as q
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression


def main():
    print("started")
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    print(df[1][1])
    # execute_regression()


def execute_regression():
    q.ApiConfig.api_key = 'Aias61x7eL8Lxwz13EWS'
    df = pd.DataFrame(q.get('WIKI/GOOGL'))
    print('columns: ', df.columns)
    print('total size: ', df.size)
    # print(df.head())
    df = df[['Adj. Close', 'Adj. Open', 'Adj. High', 'Adj. Low']]
    df['HL_PT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']
    df['OC_PT'] = (df['Adj. Open'] - df['Adj. Close']) / df['Adj. Low']
    df = df[['HL_PT', 'OC_PT', 'Adj. Close']]
    forecast_column = 'Adj. Close'
    df.fillna(-9999, inplace=True)
    part_of_data = 0.01  # percentage of data
    forecast_out = int(math.ceil(0.01 * len(df)))
    print(forecast_out)
    df['label'] = df[forecast_column].shift(-forecast_out)
    df.dropna(inplace=True)
    X = np.array(df.drop(['label'], 1))
    Y = np.array(df['label'])
    X = preprocessing.scale(X)
    # print(df.head())


if __name__ == "__main__":
    main()
