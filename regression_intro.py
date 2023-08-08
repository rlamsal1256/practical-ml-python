import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")
# print(df.head())
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
# fill in the missing data
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

x = df.drop(['label'], axis=1).to_numpy()
y = np.array(df['label'])

x = preprocessing.scale(x)

# x = x[:-forecast_out+1]
df.dropna(inplace=True)
y = np.array(df['label'])

# print(len(x), len(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = LinearRegression()
# clf = svm.SVR() # support vector regression. this is how easy it is to change the algorithm
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)
