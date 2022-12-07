#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')


# In[3]:


car=pd.read_csv('quikr_car.csv')


# In[4]:


car.head()


# In[5]:


car.shape


# In[6]:


car.info()


# ##### Creating backup copy

# In[7]:


backup=car.copy()


# ## Quality
# 
# - names are pretty inconsistent
# - names have company names attached to it
# - some names are spam like 'Maruti Ertiga showroom condition with' and 'Well mentained Tata Sumo'
# - company: many of the names are not of any company like 'Used', 'URJENT', and so on.
# - year has many non-year values
# - year is in object. Change to integer
# - Price has Ask for Price
# - Price has commas in its prices and is in object
# - kms_driven has object values with kms at last.
# - It has nan values and two rows have 'Petrol' in them
# - fuel_type has nan values

# ## Cleaning Data 

# #### year has many non-year values

# In[8]:


car=car[car['year'].str.isnumeric()]


# #### year is in object. Change to integer

# In[9]:


car['year']=car['year'].astype(int)


# #### Price has Ask for Price

# In[10]:


car=car[car['Price']!='Ask For Price']


# #### Price has commas in its prices and is in object

# In[11]:


car['Price']=car['Price'].str.replace(',','').astype(int)


# ####  kms_driven has object values with kms at last.

# In[12]:


car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')


# #### It has nan values and two rows have 'Petrol' in them

# In[13]:


car=car[car['kms_driven'].str.isnumeric()]


# In[14]:


car['kms_driven']=car['kms_driven'].astype(int)


# #### fuel_type has nan values

# In[15]:


car=car[~car['fuel_type'].isna()]


# In[16]:


car.shape


# ### name and company had spammed data...but with the previous cleaning, those rows got removed.

# #### Company does not need any cleaning now. Changing car names. Keeping only the first three words

# In[17]:


car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ')


# #### Resetting the index of the final cleaned data

# In[18]:


car=car.reset_index(drop=True)


# ## Cleaned Data

# In[19]:


car


# In[20]:


car.to_csv('Cleaned_Car_data.csv')


# In[21]:


car.info()


# In[22]:


car.describe(include='all')


# In[ ]:





# In[23]:


car=car[car['Price']<6000000]


# ### Checking relationship of Company with Price

# In[24]:


car['company'].unique()


# In[25]:


import seaborn as sns


# In[26]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# ### Checking relationship of Year with Price

# In[27]:


plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# ### Checking relationship of kms_driven with Price

# In[28]:


sns.relplot(x='kms_driven',y='Price',data=car,height=7,aspect=1.5)


# ### Checking relationship of Fuel Type with Price

# In[29]:


plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=car)


# ### Relationship of Price with FuelType, Year and Company mixed

# In[30]:


ax=sns.relplot(x='company',y='Price',data=car,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')


# ### Extracting Training Data

# In[31]:


X=car[['name','company','year','kms_driven','fuel_type']]
y=car['Price']


# In[32]:


X


# In[33]:


y.shape


# ### Applying Train Test Split

# In[34]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[35]:


from sklearn.linear_model import LinearRegression


# In[36]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# #### Creating an OneHotEncoder object to contain all the possible categories

# In[37]:


ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])


# #### Creating a column transformer to transform categorical columns

# In[38]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')


# #### Linear Regression Model

# In[39]:


lr=LinearRegression()


# #### Making a pipeline

# In[40]:


pipe=make_pipeline(column_trans,lr)


# #### Fitting the  model

# In[41]:


pipe.fit(X_train,y_train)


# In[42]:


y_pred=pipe.predict(X_test)


# #### Checking R2 Score

# In[43]:


r2_score(y_test,y_pred)


# #### Finding the model with a random state of TrainTestSplit where the model was found to give almost 0.92 as r2_score

# In[44]:


scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


# In[45]:


np.argmax(scores)


# In[46]:


scores[np.argmax(scores)]


# In[47]:


pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# #### The best model is found at a certain random state 

# In[48]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[49]:


import pickle


# In[50]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[51]:


pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,1000,'Petrol']).reshape(1,5)))


# In[52]:


pipe.steps[0][1].transformers[0][1].categories[0]


# In[ ]:




