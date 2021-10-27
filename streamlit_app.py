import streamlit as st
import pandas as pd
import numpy as np
import warnings # Supress Warnings
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split

def toTimestamp(s:str):
  return time.mktime(datetime.strptime(s, "%Y-%m-%d").timetuple())

def fromTimestamp(s):
  return datetime.fromtimestamp(s).strftime("%Y-%m-%d")


warnings.filterwarnings('ignore')

histo = pd.read_csv("./Dogecoin_history.csv", delimiter=',', header=[0])
trendsDoge = pd.read_csv("./Dogecoin_trends.csv", delimiter=',', header=[1])

index = histo.loc[histo['Date'].str.match('.*-.*-01') == False].index
histo = histo.drop(index)
histo.reset_index(inplace=True, drop=True)

trendsDoge = trendsDoge.drop(trendsDoge.loc[0:8].index)
trendsDoge.reset_index(inplace=True, drop=True)

print(histo.shape)
print(trendsDoge.shape)



dataset = pd.DataFrame()
dataset["endDayValue"] = histo["Close"]
dataset["volume"] = histo["Volume"]
dataset["date"] = histo["Date"]
dataset["trends"] = trendsDoge["Dogecoin: (Dans tous les pays)"]

timestamp = [toTimestamp(i) for i in histo["Date"].to_list()]
dataset["timestamp"] = timestamp

dataset.loc[dataset["trends"] == "<\xa01"] = 0 # Replacing the < 1 by 0
dataset["trends"] = dataset['trends'].astype(float)

feature = ["timestamp", "trends"]
target = "endDayValue"
y = dataset[target]
X = dataset[feature]

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.33, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

regFinal = LinearRegression().fit(X, y.values.ravel())

X_pred = reg.predict(X_test)

def getPredictions(x1):
    if x1 == None:
        st.write("[Error] Pick a date")
        return
    dateS = toTimestamp(str(x1))
    print("Linear Regression")
    print("----------")
    print("[IA] Let's print the possible value of Dogecoin with a given trend")
    print("----------")
    print("")
    for i in range(0, 125, 25):
        st.write(f'At [{fromTimestamp(dateS)}] and trend [{i}] the [Dogecoin value] will be {regFinal.predict([[1635689721.0, i]])[0]}')
d = st.date_input(
    "When do you want to predict (*)",
    datetime.now())

if st.button('Make predictions'):
    getPredictions(d)