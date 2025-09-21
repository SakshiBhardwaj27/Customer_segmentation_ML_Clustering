#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:green" align="center"><b> Market Segmentation in SBI life Insurance</b> </h1>

# # **1. Import Libraries:**

# In[1]:


# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score


# # **2. Load Dataset:**

# In[2]:


# import the dataset
creditcard_df = pd.read_csv("credit_card_dataset.csv")
creditcard_df.head()


# # **3.Exploratory Data Analysis & Data Cleaning:**

# In[3]:


creditcard_df.shape


# In[4]:


# information about the data
creditcard_df.info()


# In[5]:


# Check the statistics summary of the dataframe
creditcard_df.describe()


# In[6]:


# checking for Null values in data frame
creditcard_df.isnull().sum()


# In[7]:


# find all columns having missing values
missing_var = [var for var in creditcard_df.columns if creditcard_df[var].isnull().sum()>0]
missing_var


# In[8]:


# fill mean value in place of missing values
creditcard_df["MINIMUM_PAYMENTS"] = creditcard_df["MINIMUM_PAYMENTS"].fillna(creditcard_df["MINIMUM_PAYMENTS"].mean())
creditcard_df["CREDIT_LIMIT"] = creditcard_df["CREDIT_LIMIT"].fillna(creditcard_df["CREDIT_LIMIT"].mean())


# In[9]:


# Again check for null values
creditcard_df.isnull().sum()


# In[10]:


# check duplicate entries in the dataset
creditcard_df.duplicated().sum()


# In[11]:


# drop unnecessary columns
creditcard_df.drop(columns=["CUST_ID"],axis=1,inplace=True)


# In[12]:


creditcard_df.columns


# In[13]:


creditcard_df.head()


# # **4. Outlier Detection**

# In[14]:


# find outlier in all columns
for i in creditcard_df.select_dtypes(include=['float64','int64']).columns:
  max_thresold = creditcard_df[i].quantile(0.95)
  min_thresold = creditcard_df[i].quantile(0.05)
  creditcard_df_no_outlier = creditcard_df[(creditcard_df[i] < max_thresold) & (creditcard_df[i] > min_thresold)].shape
  print(" outlier in ",i,"is" ,int(((creditcard_df.shape[0]-creditcard_df_no_outlier[0])/creditcard_df.shape[0])*100),"%")


# In[15]:


# remove outliers from columns having nearly 10% outlier
max_thresold_BALANCE = creditcard_df["BALANCE"].quantile(0.95)
min_thresold_BALANCE = creditcard_df["BALANCE"].quantile(0.05)
max_thresold_CREDIT_LIMIT = creditcard_df["CREDIT_LIMIT"].quantile(0.95)
min_thresold_CREDIT_LIMIT = creditcard_df["CREDIT_LIMIT"].quantile(0.05)
max_thresold_PAYMENTS = creditcard_df["PAYMENTS"].quantile(0.95)
min_thresold_PAYMENTS = creditcard_df["PAYMENTS"].quantile(0.05)
creditcard_df_no_outlier = creditcard_df[(creditcard_df["CREDIT_LIMIT"] < max_thresold_CREDIT_LIMIT) & (creditcard_df["CREDIT_LIMIT"] > min_thresold_CREDIT_LIMIT) & (creditcard_df["BALANCE"] < max_thresold_BALANCE) & (creditcard_df["BALANCE"] > min_thresold_BALANCE) &  (creditcard_df["PAYMENTS"] < max_thresold_PAYMENTS) & (creditcard_df["PAYMENTS"] > min_thresold_PAYMENTS)]


# In[16]:


# DataFrame having no outlier
creditcard_df_no_outlier.head()


# In[17]:


creditcard_df_no_outlier.shape


# In[18]:


# correlation matrix of DataFrame
plt.figure(figsize=(20,10))
corn=creditcard_df_no_outlier.corr()
sns.heatmap(corn,annot=True,cmap="BuPu",fmt='.2f')


# ## From the results, we can see 3 pairs of strong correlation
# 1. "PURCHASES" and "ONEOFF_PURCHASES" -- 0.86
# 2. "PURCHASES_FREQUENCY" and 'PURCHASES_INSTALLMENT_FREQUENCY' --0.85
# 3. "CASH_ADVANCE_TRX" and "CASH_ADVANCE_FREQUENCY" --0.81

# # **5. Scaling the data**

# The next step is to scale our values to give them all equal importance. Scaling is also important from a clustering perspective as the distance between points affects the way clusters are formed.
# 
# Using the StandardScaler, we transform our dataframe into the following numpy arrays

# In[19]:


# scale the DataFrame
scalar=StandardScaler()
creditcard_scaled_df = scalar.fit_transform(creditcard_df_no_outlier)


# In[20]:


creditcard_scaled_df


# # **6. Dimensionality reduction**

# -> Dimensionality reduction is a technique used to reduce the number of features in a dataset while retaining as much of the important information as possible. 
# 
# -> In other words, it is a process of transforming high-dimensional data into a lower-dimensional space that still preserves the essence of the original data.
# 
# -> This can be done for a variety of reasons, such as to reduce the complexity of a model, to reduce the storage space, to improve the performance of a learning algorithm, or to make it easier to visualize the data. 
# 
# -> There are several techniques for dimensionality reduction, 
# * including principal component analysis (PCA), 
# * singular value decomposition (SVD), 
# * and linear discriminant analysis (LDA). 
# 
# Each technique uses a different method to project the data onto a lower-dimensional space while preserving important information.

# In[21]:


# convert the DataFrame into 2D DataFrame for visualization
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_scaled_df)
pca_df = pd.DataFrame(data=principal_comp,columns=["pca1","pca2"])
pca_df.head()


# # **7. Hyperparameter tuning**

# In[22]:


# find 'k' value by Elbow Method
inertia = []
range_val = range(1,15)
for i in range_val:
  kmean = KMeans(n_clusters=i)
  kmean.fit_predict(pd.DataFrame(creditcard_scaled_df))
  inertia.append(kmean.inertia_)
plt.plot(range_val,inertia,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()


# From this plot, 4th cluster seems to be the elbow of the curve.
# However, the values does not reduce to linearly until 8th cluster, so we may consider using 8 clusters in this case.

# # **8. Model Building**

# ## ** K-Means Clustering**

# In[23]:


# apply kmeans algorithm
kmeans_model=KMeans(4)
kmeans_model.fit_predict(creditcard_scaled_df)
pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)


# In[24]:


# visualize the clustered dataframe
# Scatter Plot
plt.figure(figsize=(8,8))
#palette=['dodgerblue','red','green','blue','black','pink','gray','purple','coolwarm']
ax=sns.scatterplot(x="pca1",y="pca2",hue="cluster",data=pca_df_kmeans,palette=['red','green','blue','black'])
plt.title("Clustering using K-Means Algorithm")
plt.show()


# ## **8.1. Analyzing Clustering Output**

# We've used K-Means model for clustering in this dataset.

# In[25]:


kmeans_model.cluster_centers_.shape


# In[26]:


# find all cluster centers
cluster_centers = pd.DataFrame(data=kmeans_model.cluster_centers_,columns=[creditcard_df.columns])
# inverse transfor the data
cluster_centers = scalar.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data=cluster_centers,columns=[creditcard_df.columns])
cluster_centers


# In[27]:


# create a column as "cluster" & store the respective cluster name that they belongs to
creditcard_cluster_df = pd.concat([creditcard_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)
creditcard_cluster_df.head()


# ## **8.2 Outcome**

# -> There are 4 clusters (segments)- each clusters are shown below in detail:
# * First Customers cluster (Transactors): Those are customers who pay least amount of interest charges and careful with their money, Cluster with lowest balance (104 Dollar) and cash advance (303 Dollar), Percentage of full payment = 23%
# 
# * Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance (5000 Dollar) and cash advance (5000 Dollar), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)
# 
# * Third customer cluster (VIP/Prime): high credit limit 16K Dollar and highest percentage of full payment, target for increase credit limit and increase spending habits
# 
# * Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance 

# ## **8.3. Analysis of each Cluster**

# ### Cluster - 1

# In[28]:


cluster_1_df = creditcard_cluster_df[creditcard_cluster_df["cluster"]==0]
cluster_1_df.sort_values(by=['BALANCE'], ascending=False).head()


# ### Cluster - 2

# In[29]:


cluster_2_df = creditcard_cluster_df[creditcard_cluster_df["cluster"]==1]
cluster_2_df.sort_values(by=['BALANCE'], ascending=False).head()


# ### Cluster - 3 (Silver)

# In[30]:


cluster_3_df = creditcard_cluster_df[creditcard_cluster_df["cluster"]==2]
cluster_3_df.sort_values(by=['BALANCE'], ascending=False).head()


# ### Cluster - 4

# In[31]:


cluster_4_df = creditcard_cluster_df[creditcard_cluster_df["cluster"] == 3]
cluster_4_df.sort_values(by=['BALANCE'], ascending=False).head()


# ## Optional

# # **9. Save The Model**

# In[32]:


#Saving Scikitlearn models
import joblib
joblib.dump(kmeans_model, "kmeans_model.pkl")


# In[33]:


# save the dataframe in .csv file named as "Clustered_Costumer_Data"
creditcard_cluster_df.to_csv("Clustered_Customer_Data.csv")


# In[ ]:




