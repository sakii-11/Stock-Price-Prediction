#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import yfinance as yf


# In[3]:


#pip install yfinance


# In[4]:


start = '2012-01-01'
end = '2022-12-31'
stock ='GOOG'

data= yf.download(stock,start,end)


# In[5]:


print(data.head())


# In[6]:


data.isnull()


# In[7]:


data.reset_index(inplace=True) #to set index


# In[8]:


data.head()


# In[9]:


movingAvg_100_days = data.Close.rolling(100).mean()  #to find moving avg of last 100 days


# In[10]:


print(movingAvg_100_days)


# In[11]:


plt.figure(figsize=(9,8))
plt.plot(movingAvg_100_days, 'r')
plt.plot(data.Close,'g')


# In[12]:


movingAvg_200_days = data.Close.rolling(200).mean()


# In[13]:


plt.figure(figsize=(9,8))
plt.plot(movingAvg_200_days,'r')
plt.plot(movingAvg_100_days,'b')
plt.plot(data.Close, 'g')


# # # Splitting of data set

# 80% of data is seperated into the training set while 20% is seperated into testing set 

# In[15]:


train_set = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
test_set = pd.DataFrame(data.Close[int(len(data)*0.80): int(len(data))])


# In[16]:


print(train_set)


# In[17]:


test_set


# In[18]:


train_set.shape[0]


# In[19]:


test_set.shape[0]


# In[20]:


from sklearn.preprocessing import MinMaxScaler  #used to deal with outliers
scaler = MinMaxScaler(feature_range=(0,1))#here we are using it to fit data between 0and 1


# In[21]:


data_train_scale = scaler.fit_transform(train_set)


# In[22]:


#we will be using the last 100 days data to predict the 101 day price 
#so we will split and save the last 100 days prcies in x array 
x =[]
y =[]

for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])  
    y.append(data_train_scale[i,0])
    
x,y= np.array(x), np.array(y)
    


# In[23]:


from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# In[24]:


model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True,
               input_shape= ((x.shape[1],1))))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
          
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
          
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
          
model.add(Dense(units=1))


# In[25]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[26]:


model.fit(x,y, epochs=50, batch_size=32,verbose=1 )


# In[27]:


model.summary()


# In[28]:


pas_100_days = train_set.tail(100)


# In[32]:


test_data = pd.concat([pas_100_days, test_set], ignore_index = True)


# In[33]:


test_data


# In[34]:


data_test_scale = scaler.fit_transform(test_data)


# In[35]:


x =[]
y =[]

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])  
    y.append(data_test_scale[i,0])
    
x,y= np.array(x), np.array(y)


# In[36]:


y_predict = model.predict(x)


# In[37]:


y_predict


# In[41]:


scale = 1/scaler.scale_


# In[42]:


y_predict = y_predict*scale


# In[43]:


y= y*scale


# In[44]:


plt.figure(figsize = (10,8))
plt.plot(y_predict,'r', label= 'Predicted Price')
plt.plot(y,'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[45]:


joblib.dump(model, 'Stock-Model.keras')


# In[ ]:




