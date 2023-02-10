# -*- coding: utf-8 -*-


#step 1: import Library

import pandas as pd

#step 2: import data

salary = pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Salary%20Data.csv')

salary.head()

salary.info()

salary.describe()

# step 3: define target (y) and features (x)

salary.columns

y = salary[['Salary']]
X = salary[['Experience Years']]

salary.shape

X.shape

y.shape

#step 4: split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2529)

#step 5: select model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#step 6: train
model.fit(X_train,y_train)

model.intercept_

model.coef_

#step 7: predict
y_pred=model.predict(X_test)

#step 8: error
from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(y_test,y_pred)
