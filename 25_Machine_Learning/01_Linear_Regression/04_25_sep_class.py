import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv(r"C:\Users\Test\Downloads\Datasets\Investment.csv")

x = df.iloc[:,:-1]
y = df.iloc[:,4]

x = pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split

x_train, x_test , y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

x = np.append(arr=np.full((50,1),42467).astype(int), values=x,axis=1)


# feature elimination technique
import statsmodels.api as sm

x_opt = x[:,[0,1,2,3,4,5]]
# ordinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm

x_opt = x[:,[0,1,2,3,5]]
# ordinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,2,3]]
# ordinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,3]]
# ordinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1]]
# ordinaryLeastSquares
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()