# simple linear regression Algo

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Test\.spyder-py3\Machine_Learning\Salary_Data.csv')


x = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)

print(regressor.get_params())


y_pred = regressor.predict(x_test)

print(y_pred)


comparision = pd.DataFrame({'Actual':y_test, "prediction":y_pred})
print(comparision)

plt.figure(figsize=(3,2))
plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_test, regressor.predict(x_test),color='blue')
plt.show()

#----------------------------------------------------------
# 24_sep_2026

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = (m_slope*12)+c_intercept
print(y_12)


# ---------------------------------------------------------------

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

# ---------------------------------------------------------------
# Mean

df.mean()

df['Salary'].mean()

df.median()
df['Salary'].median()


# Mode 
df['Salary'].mode()

df.var()

df['Salary'].var()

# Standard daviation

df.std()
df['Salary'].std()


# coefficient

from scipy.stats import variation

variation(df.values)


variation(df['Salary'])


# corr
df.corr()

df['Salary'].corr(df['YearsExperience'])

# skewness

df.skew()

df['Salary'].skew()

# standerd error 
df.sem()

df['Salary'].sem()


# anova == ssr , sse ,sst

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

# SSE

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# SST

mean_total = np.mean(df.values)
SST = np.sum((df.values-mean_total)**2)
print(SST)

# r2
r_square = 1-SSR/SST
print(r_square)



import pickle

file = "model.pkl"
with open(file, 'wb') as f:
    pickle.dump(regressor, f)