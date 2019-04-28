
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_squared_error


# In[15]:


#Loading the DataSet
#import os
#os.chdir("F:\PRANAV\Project\Car Rental Prediction")
df_train = pd.read_csv('train_cab.csv')
df_test = pd.read_csv('test.csv')


# In[16]:


#Converting data types to apply regression
df_train= df_train.drop(df_train.index[1123])
df_train = df_train.reset_index(drop = True)
df_train['fare_amount'] = df_train['fare_amount'].apply(pd.to_numeric)


# In[17]:


#Missing Value Analysis
missing_Val = pd.DataFrame(df_train.isnull().sum())
missing_Val = missing_Val.reset_index()
missing_Val = missing_Val.rename(columns = {'index':'Variables',0:'Missing_Percentage'})
missing_Val['Missing_Percentage'] = (missing_Val['Missing_Percentage']/len(df_train))*100
missing_Val = missing_Val.sort_values('Missing_Percentage', ascending = False).reset_index(drop=True)
df_train = df_train.fillna(df_train.median())
df_train.isnull().sum()


# In[19]:


# removing outliers if any
#q1 = df_train["fare_amount"].quantile(0.25)
#q3 = df_train["fare_amount"].quantile(0.75)
#iqr = q3-q1 #Interquartile range
#fence_low  = q1-1.5*iqr
#fence_high = q3+1.5*iqr
#df_train = df_train.loc[(df_train["fare_amount"] > fence_low) & (df_train["fare_amount"] < fence_high)]

# Another way to detect outlier via 3 standard deviation rule
len(df_train[((df_train.fare_amount - df_train.fare_amount.mean()) / df_train.fare_amount.std()).abs() < 3])


# In[20]:


# Converting categorical variables as dummies
y=df_train["fare_amount"].tolist()
df_train.drop( ['pickup_datetime','fare_amount'], axis=1, inplace=True)
df_test.drop(['pickup_datetime'], axis =1, inplace = True)
X=df_train.as_matrix()
X_test=df_test.as_matrix()


# In[21]:


# Split dataset into train-test as 80:20 split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Loading different models
clf1=RandomForestRegressor()
clf2=GradientBoostingRegressor()
clf3=LinearRegression()


# In[23]:


# Training all 3 models
clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)


# In[24]:


# RMSE for all 3 models

y_pred1=clf1.predict(X_val)
y_pred2=clf2.predict(X_val)
y_pred3=clf3.predict(X_val)

print("RMSE for Random Forest Regressor: ",sqrt(mean_squared_error(y_pred1,y_val)))
print("RMSE for Gradient Boosting Regressor: ",sqrt(mean_squared_error(y_pred2,y_val)))
print("RMSE for Linear Regression Model: ",sqrt(mean_squared_error(y_pred3,y_val)))


# In[25]:


param_grid  = {'n_estimators': [10,100,50,300], 
               'max_features': [1,3,5], 
               'max_depth':[20,20,25,25],
               'min_samples_leaf':[100]
              }
# Create a based model
rf = RandomForestRegressor()
#Instantiate the grid search model
clf = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 3)
clf.fit(X, y)


# In[13]:


y_pred=clf.predict(X_test)


# In[15]:


newDF=pd.DataFrame()
newDF["Predicted_fare"]=y_pred
newDF.to_csv("predictions.csv",index=False)

