import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv(r"C:\Users\Test\.spyder-py3\Machine_Learning\Employee_Data.csv")

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#--------------------------------------------------------

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()

x[:,1:3] = imputer.fit_transform(x[:,1:3])
print(imputer)
#--------------------------------------------------------

#from sklearn.preprocessing import OneHotEncoder
#onehot_x = OneHotEncoder()
#x[:,0] = onehot_x.fit_transform(x[:,0])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#--------------------------------------------------------

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,
                                                 random_state=0)

#--------------------------------------------------------

from sklearn.preprocessing import StandardScaler
standard = StandardScaler()

x_train = standard.fit_transform(x_train) 
x_test = standard.transform(x_test)

#--------------------------------------------------------

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)

#----------------------------------------------------------
# 24_sep_2026







