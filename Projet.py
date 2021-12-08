# %%
import pandas as pd
import numpy as np
from sklearn import model_selection
# %%
df = pd.read_csv("corporate_rating.csv")
# %%
df["RatingScore"] = df["Rating"].apply(lambda x:key[x])
# %%
NumericVariables = ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
       'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin',
       'operatingProfitMargin', 'returnOnAssets', 'returnOnCapitalEmployed',
       'returnOnEquity', 'assetTurnover', 'fixedAssetTurnover',
       'debtEquityRatio', 'debtRatio', 'effectiveTaxRate',
       'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare',
       'cashPerShare', 'companyEquityMultiplier', 'ebitPerRevenue',
       'enterpriseValueMultiple', 'operatingCashFlowPerShare',
       'operatingCashFlowSalesRatio', 'payablesTurnover']
# %%
corr = df.corr()["RatingScore"]
NumericVariables = corr[abs(corr)>=0.05].index
# %%
X_numeric = df[NumericVariables].values
# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X_numeric = scaler.fit_transform(X_numeric)
# %%
CategoricakVariables = ["Rating Agency Name", "Sector"]
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(df[CategoricakVariables]).toarray()
# %%
X = np.hstack([X_categorical, X_numeric])
Y = df['Rating'].values
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# %%
model = RandomForestClassifier()
# %%
model.fit(X_train, Y_train)
# %%
predictedRatings = model.predict(X_test)
# %%
from sklearn.metrics import accuracy_score
# %%
print(f"Accuracy score = {accuracy_score(Y_test, predictedRatings)}")
# %%
key={'D':0,'C':1,'CC':2,'CCC':3,'B':4,'BB':5,'BBB':6,'A':7,'AA':8,'AAA':9}
y = pd.get_dummies(Y)
cols = key.keys()
y = y[cols].values
# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
# %%
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(56, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='RMSprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# %%
model.fit(X_train, y_train, epochs=100)
# %%
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    
# %%
