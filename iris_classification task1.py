#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import seaborn as sns
warnings.filterwarnings(action = 'ignore', category = FutureWarning)


# In[2]:


#import the dataset
iris = pd.read_csv("iris.csv")


# # preliminary data inspection

# In[3]:


iris.shape


# In[4]:


iris.head


# In[5]:


iris.describe()


# In[14]:


#id is not an important parameter so dropping it
iris.drop('Id', axis=1 ,inplace= True)


# In[15]:


iris.head()


# In[16]:


iris.isna().sum()


# In[17]:


#checking for biases in the dataset
iris['Species'].value_counts()


# In[20]:


#visualize the different species and their count
sns.countplot(iris['Species'])
plt.xlabel('Iris Species')
plt.ylabel('Count')
plt.title("different iris flower species")


# #Dataset is balanced dataset.
# #visualizing each independent variable with respect to species using box plot

# In[23]:


plt.figure(figsize=(7,7))
sns.boxplot(x='SepalLengthCm' , y = 'Species' , data = iris)


# In[24]:


plt.figure(figsize=(7,7))
sns.boxplot(x='SepalWidthCm' , y = 'Species' , data = iris)


# In[26]:


plt.figure(figsize=(7,7))
sns.boxplot(x='PetalLengthCm' , y = 'Species' , data = iris)


# In[28]:


plt.figure(figsize=(7,7))
sns.boxplot(x='PetalWidthCm' , y = 'Species' , data = iris)


# # we can see that the petal width and petal length of iris sentosa is visibly smaller than sepal width and sepal length

# # visualizing the whole dataset using pairplot

# In[30]:


sns.pairplot(iris, hue='Species')


# # using different models and checking the accuracy 1.logistic regression 2.random forest 3.svm

# In[31]:


#Importing Logistic Regression model
from sklearn.linear_model import LogisticRegression


# In[32]:


lr= LogisticRegression()


# In[35]:


#Training and Testing data

X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = iris['Species']

#Importing "train_test-split" function to test the model
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)


# In[36]:


X_train.head()


# In[34]:


print("Train dataset shape is", X_train.shape)
print("Test dataset shape is", X_test.shape)


# In[37]:


#Fit the model in train and test data
lr.fit(X_train,y_train).score(X_train,y_train)


# In[38]:


#Now fitting the model in test set
prediction=lr.predict(X_test)


# In[39]:


#Printing first 5 rows after fitting the model in test set
print (X_test.head())


# In[40]:


from sklearn import metrics
cm = metrics.confusion_matrix(y_test, prediction) 
print(cm)
accuracy = metrics.accuracy_score(y_test, prediction) 
print("Accuracy score:",accuracy)


# # so the logistic regression gives the 95.55% of accuracy

# # 2.random forest

# In[41]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)


# In[42]:


predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# # random forest gives the 96% of accuracy

# # 3.SVM

# In[43]:


#SVM
from sklearn.svm import SVC


# In[44]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
iris['Species']=encoder.fit_transform(iris['Species'])


# In[45]:


model=SVC()
model.fit(X_train, y_train)


# In[46]:


#creating the predictions on the test data i.e. X_test
y_pred =model.predict(X_test)
y_pred


# In[47]:


from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)
print("Accuracy for the SVM Classifier: {:.3f}".format(accuracy)) #.3f is upto 3 places decimal


# # So the SVM gives 95.6% accuracy .#RANDOM FOREST IS PERFORMING BETTER THAN OTHER TWO.

# In[ ]:




