import streamlit as st
import pandas as pd
import numpy as np
import warnings # Supress Warnings
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="[Machine-Learning]", page_icon="📈", layout='centered', initial_sidebar_state='auto', menu_items=None)
def toTimestamp(s:str):
  return time.mktime(datetime.strptime(s, "%Y-%m-%d").timetuple())

def fromTimestamp(s):
  return datetime.fromtimestamp(s).strftime("%Y-%m-%d")


st.title('[Machine-Learning] Dogecoin')

st.write("Welcome to the Dogecoin prediction web-app .")
st.write("We harvested data from Google Trends to get Dogecoin's trends since 2014 to october 2021.")
st.write("We also gather the Dogecoin history usd value since 2014 to october 2021.")
st.write("We was then able to make some machine learning in order to predict the Dogecoin value at a given date.")
st.write("⚠️ This is a school project, these predictions should be taken with a grain of salt ⚠️")

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

df2 = pd.DataFrame()
df2[''] = dataset['endDayValue']
df3 = pd.DataFrame()
df3[''] = dataset['trends']

col1, col2 = st.columns([1, 1])
col1.subheader("Dogecoin USD value")
col1.line_chart(df2, width=300)
col2.subheader("Dogecoin Trends value")
col2.line_chart(df3, width=300) # Trend
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
    #for i in range(0, 125, 25):
    #    st.write(f'At [{fromTimestamp(dateS)}] and trend [{i}] the [Dogecoin value] will be {regFinal.predict([[dateS, i]])[0]}')
    meanArr = np.array([regFinal.predict([[dateS, i]])[0] for i in range(0, 125, 25)])
    dateArr = np.array([ fromTimestamp(dateS) for i in range(0, 125, 25) ])
    trend = np.array([i for i in range(0, 125, 25) ])

    df1 = pd.DataFrame()
    df1["predictionValue"] = meanArr
    df1["trend"] = trend
    st.dataframe(df1)
    print(meanArr.shape)
    print(dateArr.shape)
    roundMean = str(round(np.mean(meanArr), 4))
    st.markdown("On the date of **<font color='#34c924'>" + fromTimestamp(dateS) + "</font>**, there is <font color='#fde368'>**50%**</font> chance that the dogecoin value will be > to ** <font color='#34c924'>" + roundMean+  "$</font>**", unsafe_allow_html=True)
    st.write()

d = st.date_input(
    "When do you want to predict (*)",
    datetime.now())

if st.button('Make predictions'):
    getPredictions(d)