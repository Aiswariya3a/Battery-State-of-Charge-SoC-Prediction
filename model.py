
import kagglehub
atechnohazard_battery_and_heating_data_in_real_driving_cycles_path = kagglehub.dataset_download('atechnohazard/battery-and-heating-data-in-real-driving-cycles')

print('Data source import complete.')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib


inputTrip = "/root/.cache/kagglehub/datasets/atechnohazard/battery-and-heating-data-in-real-driving-cycles/versions/1/TripA01"
dataTrip1 = pd.read_csv(f"{inputTrip}.csv",sep=";", encoding='unicode_escape')
dfTrip1   = pd.DataFrame(dataTrip1)

def conform_datasets(nbOfTrips, pathToFiles, col_list=None):
    'Function that looks through the different trips to see if they all'
    'contain the same features. If not, remove those that are not present'
    'everywhere. Otherwise training and inference may not work properly'

    dfSummerTrips = []
    if not col_list:
        col_list = []
    inputTripsA = pathToFiles
    for i in range(0,nbOfTrips):
        dataTrip = pd.read_csv(f"{inputTripsA}{(i+1):02}.csv", sep=";", encoding='unicode_escape')
        dfSummerTrips.append(pd.DataFrame(dataTrip))
        if not col_list:
            col_list = dfSummerTrips[i].columns.tolist()
        elif set(col_list)!= set(dfSummerTrips[i].columns.tolist()):
            diff = list(set(col_list)-set(dfSummerTrips[i].columns.tolist()))
            for item in diff:
                print(f"Discrepancy with Trip: {(i+1):02}")
                print(f"Removed >>{item}<< from columns list since it is not consistently present for all trips")
                col_list.remove(item)

    for i, trip in enumerate(dfSummerTrips):
        dfSummerTrips[i] = trip[col_list]
    return dfSummerTrips, col_list

pathA = "/root/.cache/kagglehub/datasets/atechnohazard/battery-and-heating-data-in-real-driving-cycles/versions/1/TripA"
dfSummerTrips, consistent_cols = conform_datasets(nbOfTrips=32,pathToFiles=pathA)


print(consistent_cols)
consistent_cols.remove("min. SoC [%]")
consistent_cols.remove("max. SoC [%)")
consistent_cols.remove("max. Battery Temperature [°C]")
dfSummerTrips, consistent_cols = conform_datasets(nbOfTrips=32,pathToFiles=pathA,col_list=consistent_cols)

def prepare_features(df_TripList,stepWidth, dropNan=False):
    newfeatures = []

    for i, trip_df in enumerate(df_TripList):

        if dropNan:
            trip_df = trip_df.dropna()
            trip_df.index = range(len(trip_df))

        dfLength = len(trip_df.index)
        numRows  = dfLength//stepWidth
        for i in range(0,numRows):
            #-Sum the actions during a given time window (makes no  practical sense to do inference every 100ms in my view)
            averageVals = trip_df.iloc[i*stepWidth:(i+1)*stepWidth].sum()
            averageVals=averageVals/(1.*stepWidth)
            featureList=averageVals.add_prefix('avrg_', axis=0)
            #-Add elevation change in the time window (absolute elevation values are not useful)
            elevationDiff = trip_df.loc[(i+1)*stepWidth,'Elevation [m]'] - trip_df.loc[i*stepWidth,'Elevation [m]']
            featureList['Elevation change']= elevationDiff
            #-Add feature values from the start of your inference windows (you want to know how the SOC changes compared to a start value)
            featuresAtBeginningOfWindow = trip_df.loc[i*stepWidth, ['SoC [%]', 'displayed SoC [%]']]
            featureList["Previous SoC"]           = featuresAtBeginningOfWindow['SoC [%]']
            featureList["Previous displayed SoC"] = featuresAtBeginningOfWindow['displayed SoC [%]']
            #-Add feature values from the end of your inference windows -> These are the values you want to predict!
            featuresAtEndOfWindow = trip_df.loc[(i+1)*stepWidth, ['SoC [%]', 'displayed SoC [%]']]
            featureList["Next SoC"]           = featuresAtEndOfWindow['SoC [%]']
            featureList["Next displayed SoC"] = featuresAtEndOfWindow['displayed SoC [%]']
            newfeatures.append(featureList)

    newDf = pd.DataFrame(newfeatures)
    newDf = newDf.drop(columns=['avrg_SoC [%]', 'avrg_displayed SoC [%]', 'avrg_Time [s]','avrg_Elevation [m]'])

    #-Check for NaN
    hasNaN = newDf.isnull().values.any()
    print(f"Output df has NaN: {hasNaN}")

    return newDf

newDf = prepare_features(dfSummerTrips, stepWidth=60)

def split_TestTrain(df, scaler=None):
    try:
        X = df.drop(columns=['Next SoC', 'Previous displayed SoC', 'Next displayed SoC'])
    except:
        X = df
    y = df['Next SoC'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if not scaler:
        scaler = StandardScaler().fit(X)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return scaler, [X_train_scaled,y_train, X_train.index], [X_test_scaled, y_test,X_test.index]

scalerSummerTrips, train_tuple, test_tuple = split_TestTrain(newDf)
X_train_scaled, y_train, index_train = train_tuple
X_test_scaled,  y_test,  index_test  = test_tuple

def fit_model_XGBoost(X,Y):
    model = XGBRegressor(n_estimators=50, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    model.fit(X, Y)
    return model

XGBoost_model = fit_model_XGBoost(X_train_scaled,y_train)
# Save the trained XGBoost model
XGBoost_model.save_model('model.json')  # or .model format

print("Model saved as 'model.json'")

newDf.to_csv("processed_dataset.csv", index=False)
print("Processed dataset saved as processed_dataset.csv")

# Assuming you have your training data in X_train
scaler = StandardScaler()
scaler.fit(X_train_scaled)  # Fit with your training data

# Save the fitted scaler
joblib.dump(scaler, 'scaler.save')