#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
data = pd.read_csv("Employee.csv")
data


# In[41]:


data.head()


# In[42]:


data.isnull().sum()


# In[43]:


data["left"].value_counts()


# In[44]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[45]:


data["salary"] = le.fit_transform(data["salary"])
data.head()


# In[46]:


x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()


# In[47]:


y=data["left"]
y.head()


# In[48]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)


# In[49]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


# In[50]:


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


# In[51]:


dt.predict([[0.5,0.8,9,260,6,0,1,2]])
print("Name:Hashwatha M")
print("Reg no:212223240051")


# In[ ]:




