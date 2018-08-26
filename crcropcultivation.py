
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt


# In[84]:


dataset = pd.read_csv("c://users/lenova/desktop/crops.csv")
dataset


# In[96]:


data1 = dataset.iloc[:,2:].values
data2 = dataset.iloc[:,1]
label = dataset.iloc[:,0].values


# In[97]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()


# In[98]:


for i in range(1,2):
    data[:,i] = labelencoder.fit_transform(data[:,i])
data2=data[:,1:2]
data2.shape


# In[99]:


from sklearn.preprocessing import Normalizer


# In[100]:


data1=data[:,1:]
normalized_data = Normalizer().fit_transform(data1)


# In[101]:


normalized_data
data2=data[:,1:2]
data2.shape


# In[102]:


df1 = np.append(normalized_data,data2,axis=1)
df1


# In[103]:


X1 = pd.DataFrame(df1,columns=['minitemp','max temp','minwater con','maxwatercon'])
X1.head(24)


# In[104]:


label = labelencoder.fit_transform(label)
print(len(label))


# In[105]:


y=pd.DataFrame(label,columns=["Suggested seasons"])
y.head(24)


# In[106]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score


# In[121]:


X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.1,random_state=40)


# In[122]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


# In[123]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[124]:


y_pred = clf.predict(X_test)


# In[125]:


y_pred


# In[126]:


cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)


# In[127]:


print("confusion matrics=",cm)
print("  ")
print("accuracy=",accuracy*100)

