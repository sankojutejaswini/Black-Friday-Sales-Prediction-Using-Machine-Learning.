#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# In[2]:


data = pd.read_csv("C:\\Users\\sanko\\Downloads\\black friday sales\\Black-Friday-Sales-Prediction-master\\Data\\BlackFridaySales.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[ ]:


CHECKING NULL VALUES


# In[6]:


data.isnull().sum()


# In[ ]:


NULL VALUE IN PERCENTAGE


# In[7]:


data.isnull().sum()/data.shape[0]*100


# In[ ]:


UNIQUE ELEMENTS IN EACH ATTRIBUTES


# In[8]:


data.nunique()


# In[ ]:


EDA


# In[ ]:


## Target Variable Purchase


# In[9]:


sns.histplot(data["Purchase"],color='r')
plt.title("Purchase Distribution")
plt.show()


# In[10]:


sns.boxplot(data["Purchase"])
plt.title("Boxplot of Purchase")
plt.show()


# In[11]:


data["Purchase"].skew()


# In[12]:


data["Purchase"].kurtosis()


# In[13]:


data["Purchase"].describe()


# In[ ]:


GENDER


# In[14]:


sns.countplot(data['Gender'])
plt.show()


# In[15]:


data['Gender'].value_counts(normalize=True)*100


# In[16]:


data.groupby("Gender").mean()["Purchase"]


# In[ ]:


MARITAL STATUS


# In[17]:


sns.countplot(data['Marital_Status'])
plt.show()


# In[18]:


data.groupby("Marital_Status").mean()["Purchase"]


# In[19]:


data.groupby("Marital_Status").mean()["Purchase"].plot(kind='bar')
plt.title("Marital_Status and Purchase Analysis")
plt.show()


# In[ ]:


OCCUPATION


# In[20]:


plt.figure(figsize=(18,5))
sns.countplot(data['Occupation'])
plt.show()


# In[21]:


occup = pd.DataFrame(data.groupby("Occupation").mean()["Purchase"])
occup


# In[22]:


occup.plot(kind='bar',figsize=(15,5))
plt.title("Occupation and Purchase Analysis")
plt.show()


# In[ ]:


CITY_CATEGORY


# In[23]:


sns.countplot(data['City_Category'])
plt.show()


# In[24]:


data.groupby("City_Category").mean()["Purchase"].plot(kind='bar')
plt.title("City Category and Purchase Analysis")
plt.show()


# In[ ]:


### Stay_In_Current_City_Years


# In[25]:


sns.countplot(data['Stay_In_Current_City_Years'])
plt.show()


# In[26]:


data.groupby("Stay_In_Current_City_Years").mean()["Purchase"].plot(kind='bar')
plt.title("Stay_In_Current_City_Years and Purchase Analysis")
plt.show()


# In[ ]:


AGE


# In[27]:


sns.countplot(data['Age'])
plt.title('Distribution of Age')
plt.xlabel('Different Categories of Age')
plt.show()


# In[28]:


data.groupby("Age").mean()["Purchase"].plot(kind='bar')


# In[29]:


data.groupby("Age").sum()['Purchase'].plot(kind="bar")
plt.title("Age and Purchase Analysis")
plt.show()


# In[ ]:


### Product_Category_1


# In[30]:


plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_1'])
plt.show()


# In[31]:


data.groupby('Product_Category_1').mean()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Mean Analysis")
plt.show()


# In[32]:


data.groupby('Product_Category_1').sum()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Analysis")
plt.show()


# In[ ]:


### Product_Category_2


# In[33]:


plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_2'])
plt.show()


# In[ ]:


### Product_Category_3


# In[34]:


plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_3'])
plt.show()


# In[35]:


data.corr()


# In[ ]:


HEATMAP


# In[36]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# In[37]:


data.columns


# In[38]:


df = data.copy()


# In[39]:


df.head()


# In[ ]:


# df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace(to_replace="4+",value="4")


# In[40]:


#Dummy Variables:
df = pd.get_dummies(df, columns=['Stay_In_Current_City_Years'])


# In[ ]:


## Encoding the categorical variables


# In[41]:


from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()


# In[42]:


df['Gender'] = lr.fit_transform(df['Gender'])


# In[43]:


df['Age'] = lr.fit_transform(df['Age'])


# In[44]:


df['City_Category'] = lr.fit_transform(df['City_Category'])


# In[45]:


df.head()


# In[46]:


df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')


# In[47]:


df.isnull().sum()


# In[48]:


df.info()


# In[ ]:


## Dropping the irrelevant columns


# In[49]:


df = df.drop(["User_ID","Product_ID"],axis=1)


# In[ ]:


## Splitting data into independent and dependent variables


# In[50]:


X = df.drop("Purchase",axis=1)


# In[51]:


y=df['Purchase']


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# In[ ]:


## Modeling


# In[ ]:


# Random Forest Regressor


# In[53]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
# create a regressor object 
RFregressor = RandomForestRegressor(random_state = 0)  


# In[54]:


RFregressor.fit(X_train, y_train)


# In[55]:


rf_y_pred = RFregressor.predict(X_test)


# In[56]:


mean_absolute_error(y_test, rf_y_pred)


# In[57]:


mean_squared_error(y_test, rf_y_pred)


# In[58]:


r2_score(y_test, rf_y_pred)


# In[ ]:


# XGBoost Regressor


# In[59]:


get_ipython().system('pip install xgboost')


# In[60]:


from xgboost.sklearn import XGBRegressor


# In[61]:


xgb_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)

xgb_reg.fit(X_train, y_train)


# In[62]:


xgb_y_pred = xgb_reg.predict(X_test)


# In[63]:


mean_absolute_error(y_test, xgb_y_pred)


# In[64]:


mean_squared_error(y_test, xgb_y_pred)


# In[65]:


r2_score(y_test, xgb_y_pred)


# In[66]:


from math import sqrt
print("RMSE of XGBoost Model is ",sqrt(mean_squared_error(y_test, xgb_y_pred)))

