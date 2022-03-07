#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import library
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import datetime as dt
from datetime import datetime
from mlxtend.frequent_patterns import apriori

# Visualisation
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import os

# Forecasting     

from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

rcParams['figure.figsize'] = 10, 6

import warnings
warnings.filterwarnings('ignore')


# In[10]:


data = pd.read_csv("Dataset_Mizan.csv")
data.head()


# In[11]:


#mendeskripsikan data yang ada pada dataset 
print(data.info())
#mencari nilai statistik penyebaran nya
print(data.describe())
print(data.shape)
print(data.isnull().sum())


# In[12]:


data.shape


# In[13]:


data["Nama_Donatur"] = data["Nama_Donatur"].fillna("Hamba Allah")
data["muzaki"] = data["muzaki"].fillna("Unknown")
data["Program"] = data["Program"].fillna("No Category")


# In[14]:


print(data.isnull().sum())


# In[15]:


data.head()


# In[16]:


print("Summary..")
#exploring the unique values of each attribute
print("Jumlah Kantor Pelayanan: ", data['Kantor_Pelayanan'].nunique())
print("Jumlah Metode Pembayaran: ", data['Metode_Pembayaran'].nunique())
print("Jumlah Program: ", data['Program'].nunique())
print("Jumlah Akad: ", data['Akad'].nunique())
print("Jumlah Donasi: ", data['ID_Donasi'].nunique())
print("Jumlah ID Donatur:", data['ID_Donatur'].nunique() )
print("Persentase donatur NA: ", round(data['ID_Donatur'].isnull().sum() * 100 / len(data),2),"%" )


# In[17]:


#tanggal terakhir di dataset
data['Tanggal'].max()


# In[18]:


now = dt.date(2022,1,17)
print(now)


# In[19]:


#membuat kolom baru yang isinya hanya tanggal
data['Tanggal_'] = pd.DatetimeIndex(data['Tanggal']).date


# In[20]:


data.head()


# In[21]:


#ID donatur dan kapan terakhir melakukan donasi
recency_df = data.groupby(by='ID_Donatur', as_index=False)['Tanggal_'].max()
recency_df.columns = ['ID_Donatur','Tgl_Terakhir_Donasi']
recency_df.head()


# In[22]:


#Menghitung recency
recency_df['Recency'] = recency_df['Tgl_Terakhir_Donasi'].apply(lambda x: (now - x).days)
recency_df.head()


# In[23]:


#drop Tgl_Terakhir_Donasi karena kita tidak gunakan lagi
recency_df.drop('Tgl_Terakhir_Donasi',axis=1,inplace=True)


# In[24]:


recency_df.head()


# In[25]:


data=data.drop_duplicates()
data.head()


# In[26]:


data.shape


# In[27]:


#menghitung frekuensi donasi
frequency_df = data.groupby(by=['ID_Donatur'], as_index=False)['ID_Donasi'].count()
frequency_df.columns = ['ID_Donatur','Frequency']
frequency_df.head()


# In[28]:


monetary_df = data.groupby(by='ID_Donatur',as_index=False).agg({'Nominal_Donasi': 'sum'})
monetary_df.columns = ['ID_Donatur','Monetary']
monetary_df.head()


# In[29]:


#menggabungkan recency_df dan frekuensi_df
temp_df = recency_df.merge(frequency_df,on='ID_Donatur')
temp_df.head()


# In[30]:


#mmenggabungkan dengan monetary_df untuk mendapatkan table dengan 3 kolom
rfm_df = temp_df.merge(monetary_df,on='ID_Donatur')
#menggunakan ID_Donatur sebagain index
rfm_df.set_index('ID_Donatur',inplace=True)
rfm_df.head()


# In[31]:


data[data['ID_Donatur']=='00019858-2827-E3C0-7EC5-6359A76A82B5']


# In[32]:


(now - dt.date(2020,6,26)).days == 570


# In[33]:


quantiles = rfm_df.quantile(q=[0.25,0.5,0.75])
quantiles


# In[34]:


quantiles.to_dict()


# In[35]:


# x = value, p = recency, monetary_value, frequency, d = quartiles dict

def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

# x = value, p = recency, monetary_value, frequency, k = quartiles dict

def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4


# In[36]:


#membuat tabel RFM segmentation 
rfm_segmentation = rfm_df
rfm_segmentation['R_Quartile'] = rfm_segmentation['Recency'].apply(RScore, args=('Recency',quantiles,))
rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,))
rfm_segmentation.head()


# In[37]:


rfm_segmentation['RFMScore'] = rfm_segmentation.R_Quartile.map(str)                             + rfm_segmentation.F_Quartile.map(str)                             + rfm_segmentation.M_Quartile.map(str)
rfm_segmentation.head()


# In[38]:


rfm_segmentation[rfm_segmentation['RFMScore']=='444'].sort_values('Monetary', ascending=False).head(10)


# In[39]:


print("Best Customers: ",len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']))
print('Loyal Customers: ',len(rfm_segmentation[rfm_segmentation['F_Quartile']==4]))
print("Big Spenders: ",len(rfm_segmentation[rfm_segmentation['M_Quartile']==4]))
print('Almost Lost: ', len(rfm_segmentation[rfm_segmentation['RFMScore']=='244']))
print('Lost Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='144']))
print('Lost Cheap Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='111']))


# In[40]:


rfm_segmentation.sort_values('RFMScore', ascending=False)


# In[41]:


rfm_segmentation[rfm_segmentation["RFMScore"] == "444"].head()


# In[42]:


rfm_segmentation[rfm_segmentation["RFMScore"] == "111"].head()


# In[43]:


rfm_segmentation.head(
)


# In[44]:


# Calculate RFM_Score
rfm_segmentation['RFM_Score'] = rfm_segmentation[['R_Quartile','F_Quartile','M_Quartile']].sum(axis=1)
print(rfm_segmentation['RFM_Score'].head())


# In[45]:


# Define rfm_level function
def rfm_level(df):
    if df['RFM_Score'] >= 9:
        return 'Can\'t Loose Them'
    elif ((df['RFM_Score'] >= 8) and (df['RFM_Score'] < 9)):
        return 'Champions'
    elif ((df['RFM_Score'] >= 7) and (df['RFM_Score'] < 8)):
        return 'Loyal'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 7)):
        return 'Potential'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 6)):
        return 'Promising'
    elif ((df['RFM_Score'] >= 4) and (df['RFM_Score'] < 5)):
        return 'Needs Attention'
    else:
        return 'Require Activation'
# Create a new variable RFM_Level
rfm_segmentation['RFM_Level'] = rfm_segmentation.apply(rfm_level, axis=1)
# Print the header with top 5 rows to the console
rfm_segmentation.head()


# In[46]:


rfm_segmentation.head()


# In[47]:


data1=data
data1.head()


# In[48]:


data2=pd.merge(left=data, right=rfm_segmentation, on="ID_Donatur", how="outer")
data2.head()


# In[49]:


data2.drop(['R_Quartile','F_Quartile','M_Quartile','Recency','Frequency','Monetary'],axis=1,inplace=True)


# In[50]:


data2.head()


# In[51]:


print(data2.shape)


# In[52]:


data2.drop(['Nama_Donatur','muzaki','Tanggal_','RFMScore','RFM_Score'],axis=1,inplace=True)


# In[53]:


data2.head()


# In[54]:


print(data2.shape)


# In[55]:


data2.isna().sum()


# In[56]:


plt.pie(data2.RFM_Level.value_counts(),
        labels=data2.RFM_Level.value_counts().index,
        autopct='%.0f%%')
plt.show()


# In[57]:


data2.reset_index(inplace=True)


# In[58]:


RFM_merge = data2[['ID_Donatur','RFM_Level']]


# In[59]:


RFM_merge.head(4)


# In[60]:


data.head(3)


# In[61]:


data.drop(['Tanggal_'],axis=1,inplace=True)


# In[62]:


Data_RFM = data.merge(RFM_merge, how= 'left', on= 'ID_Donatur')


# In[63]:


Data_RFM.head(3)


# In[64]:


Data_RFM.drop(['Nama_Donatur','muzaki'],axis=1,inplace=True)


# In[65]:


Data_RFM.isna().sum()


# In[73]:


print(data2.shape)


# In[74]:


data2.head()


# In[67]:


Data_RFM.drop_duplicates(keep=False, inplace=True)


# In[71]:


print(Data_RFM.shape)


# In[69]:


Data_RFM.head(3)


# In[6]:


# Data_RFM.to_csv('FINAL DATASET.csv')
# files.download('FINAL DATASET.csv')


# In[7]:


# arima_fc.to_csv("C:/Users/Hp/Desktop/Arima_FC.csv", index=False)


# In[75]:


data2.to_csv("C:/Users/Hp/Desktop/Data_RFM.csv", index=False)


# In[8]:


# df.to_csv('nama file', index=False)

