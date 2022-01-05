from datetime import timedelta

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings("ignore")
import os

ROOT_DIR = os.path.dirname(os.path.abspath("uploaded"))


def calculateRMSE(actual_values, predicted_values):
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    return rmse


def timeinterval(csvfile, nadditions):
    data = read_csv(ROOT_DIR + "\\app\\static\\uploaded\\" + csvfile, header=0)
    data.rename(columns={list(data)[0]: 'date'}, inplace=True)
    convert_to_datetime(data)
    data = data.set_index('date')
    firstdate = data.index[0]
    seconddate = data.index[1]
    diff = seconddate - firstdate
    print(firstdate)
    print(seconddate)
    print("Difference as days: ", diff)
    print(data)
    lastdate = data.index[-1]
    print("last date: ", lastdate)
    listofdates = []
    for index in range(nadditions):
        added = lastdate + diff
        listofdates.append(added)
        lastdate = listofdates[index]
    return listofdates


def transform_to_supervised(data, previous_steps=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(previous_steps, 0, -1):
        cols.append(df.shift(i))

    for i in range(0, n_out):
        cols.append(df.shift(-i))

    agg = pd.concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg.values


def convert_to_datetime(df):
    try:
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y/%d/%m')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%d/%Y/%m')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%m/%Y/%d')
    except:
        pass

'''
data = read_csv("C:/Users/admin/Desktop/login/app/static/uploaded/DailyDelhiClimate.csv", header=0)
convert_to_datetime(data)
values = data.values
data = data.set_index('date')


x = transform_to_supervised(data, previous_steps=3)

trainX, trainy = x[:, :-1], x[:, -1]

new_trainX = []


for elt in trainX:
    column_number = values.shape[1]-1
    elt_arranged = []
    for col_num in range(column_number):
        elt_arranged.append([elt[index] for index in [0+(col_num*column_number), 1+(col_num*column_number), 2+(col_num*column_number)]])
    elt_arranged = np.array(elt_arranged)
    new_trainX.append(elt_arranged.flatten())
new_trainX = np.array(new_trainX)


multiple_Y = []
counter_for_first_element = 0
for elt in new_trainX:
    if(counter_for_first_element>3):
        next_y = elt[0:3]
        multiple_Y.append(next_y)
    else:
        counter_for_first_element += 1
multiple_Y = np.array(multiple_Y)
'''

def forecast_algorithm(model, future_day_amount, csvfilename):
    data = read_csv(ROOT_DIR + "\\app\\static\\uploaded\\" + csvfilename, header=0)
    ######
    convert_to_datetime(data)
    values = data.values
    data = data.set_index('date')

    x = transform_to_supervised(data, previous_steps=3)

    trainX, trainy = x[:, :-1], x[:, -1]

    new_trainX = []

    for elt in trainX:
        column_number = values.shape[1] - 1
        elt_arranged = []
        for col_num in range(column_number):
            elt_arranged.append([elt[index] for index in [0 + (col_num * column_number), 1 + (col_num * column_number),
                                                          2 + (col_num * column_number)]])
        elt_arranged = np.array(elt_arranged)
        new_trainX.append(elt_arranged.flatten())
    new_trainX = np.array(new_trainX)

    multiple_Y = []
    counter_for_first_element = 0
    for elt in new_trainX:
        if (counter_for_first_element > 3):
            next_y = elt[0:3]
            multiple_Y.append(next_y)
        else:
            counter_for_first_element += 1
    multiple_Y = np.array(multiple_Y)
    ######
    '''
    convert_to_datetime(data)
    values = data.values
    data = data.set_index('date')
    '''
    row = trainX[len(trainX)-1]
    forecasted_values = []
    x_row = asarray([x[len(x)-1]])[0]

    if(model == "rfr"):
        model = RandomForestRegressor(n_estimators=100, max_depth=6)
        clf = MultiOutputRegressor(model).fit(x[:-4, :], multiple_Y)
        model.fit(trainX, trainy)

        for i in range(future_day_amount):
            multi_output = clf.predict([x_row])
            yhat = model.predict(asarray([row]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            forecasted_values.append(float(yhat))

        error_model = RandomForestRegressor(n_estimators=100, max_depth=6)
        clf_error = MultiOutputRegressor(error_model).fit(x[:int(len(x) * 0.80) - 5, :],
                                                          multiple_Y[0:int(len(multiple_Y) * 0.8) - 2])
        error_model.fit(trainX[0:int(len(x) * 0.8) - 5], trainy[0:int(len(multiple_Y) * 0.8) - 2])
        preds = []

        for i in range(int(len(x) * 0.2)):
            multi_output = clf_error.predict([asarray([x[int(len(x) * 0.80) + i]])[0]])
            yhat = error_model.predict(asarray([trainX[int(len(trainX) * 0.80) + i]]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            preds.append(float(yhat))

        rmse_error = calculateRMSE(trainy[int(len(trainy) * 0.8) + 1:], preds)

    # decision tree
    if (model == "dtr"):
        dt_reg = DecisionTreeRegressor(max_depth=22, max_features="auto", random_state=10)
        clf = MultiOutputRegressor(dt_reg).fit(x[:-4, :], multiple_Y)
        dt_reg.fit(trainX, trainy)

        for i in range(future_day_amount):
            multi_output = clf.predict([x_row])
            yhat = dt_reg.predict(asarray([row]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            forecasted_values.append(float(yhat))

        error_model = DecisionTreeRegressor(max_depth=22, max_features="auto", random_state=10)
        clf_error = MultiOutputRegressor(error_model).fit(x[:int(len(x) * 0.80) - 5, :],
                                                          multiple_Y[0:int(len(multiple_Y) * 0.8) - 2])
        error_model.fit(trainX[0:int(len(x) * 0.8) - 5], trainy[0:int(len(multiple_Y) * 0.8) - 2])
        preds = []

        for i in range(int(len(x) * 0.2)):
            multi_output = clf_error.predict([asarray([x[int(len(x) * 0.80) + i]])[0]])
            yhat = error_model.predict(asarray([trainX[int(len(trainX) * 0.80) + i]]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            preds.append(float(yhat))

        rmse_error = calculateRMSE(trainy[int(len(trainy) * 0.8) + 1:], preds)

    if(model =="xgb"):
        regressor = xgb.XGBRegressor(
                    n_estimators=100,
                    reg_lambda=1,
                    gamma=0,
                    max_depth=3
                )
        clf = MultiOutputRegressor(regressor).fit(x[:-4, :], multiple_Y)
        regressor.fit(trainX, trainy)

        for i in range(future_day_amount):
            multi_output = clf.predict([x_row])
            yhat = regressor.predict(asarray([row]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            forecasted_values.append(float(yhat))

        error_model = xgb.XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3
        )
        clf_error = MultiOutputRegressor(error_model).fit(x[:int(len(x) * 0.80) - 5, :],
                                                          multiple_Y[0:int(len(multiple_Y) * 0.8) - 2])
        error_model.fit(trainX[0:int(len(x) * 0.8) - 5], trainy[0:int(len(multiple_Y) * 0.8) - 2])
        preds = []

        for i in range(int(len(x) * 0.2)):
            multi_output = clf_error.predict([asarray([x[int(len(x) * 0.80) + i]])[0]])
            yhat = error_model.predict(asarray([trainX[int(len(trainX) * 0.80) + i]]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            preds.append(float(yhat))

        rmse_error = calculateRMSE(trainy[int(len(trainy) * 0.8) + 1:], preds)

    if (model == "svr"):
        svr_rbf = SVR(kernel='poly', C=1e2, gamma="scale")
        clf = MultiOutputRegressor(svr_rbf).fit(x[:-4, :], multiple_Y)
        svr_rbf.fit(trainX, trainy)

        for i in range(future_day_amount):
            multi_output = clf.predict([x_row])
            yhat = svr_rbf.predict(asarray([row]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            forecasted_values.append(float(yhat))

        error_model = SVR(kernel='poly', C=1e2, gamma="scale")
        clf_error = MultiOutputRegressor(error_model).fit(x[:int(len(x) * 0.80) - 5, :],
                                                          multiple_Y[0:int(len(multiple_Y) * 0.8) - 2])
        error_model.fit(trainX[0:int(len(x) * 0.8) - 5], trainy[0:int(len(multiple_Y) * 0.8) - 2])
        preds = []

        for i in range(int(len(x) * 0.2)):
            multi_output = clf_error.predict([asarray([x[int(len(x) * 0.80) + i]])[0]])
            yhat = error_model.predict(asarray([trainX[int(len(trainX) * 0.80) + i]]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            preds.append(float(yhat))

        rmse_error = calculateRMSE(trainy[int(len(trainy) * 0.8) + 1:], preds)

    return forecasted_values, rmse_error

