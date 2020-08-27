#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data=pd.read_csv('E:/Projects/heart-disease-uci/heart.csv')


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data['thal'].value_counts()


# In[9]:


data['sex'].value_counts() # 1-male and 0-female


# ### Infromation of each feature:

# age = age in years
# 
# sex =(1 = male; 0 = female)
# 
# cp = chest pain type
# 
# trestbps = resting blood pressure (in mm Hg on admission to the hospital)
# 
# chol = serum cholestoral in mg/dl
# 
# fbs = (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
# 
# restecg = resting electrocardiographic results
# 
# thalach = maximum heart rate achieved
# 
# exang = exercise induced angina (1 = yes; 0 = no)
# 
# oldpeak = ST depression induced by exercise relative to rest
# 
# slope = the slope of the peak exercise ST segment
# 
# ca = number of major vessels (0-3) colored by flourosopy
# 
# thal  = 3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# target = 1 or 0

# In[183]:


data.describe()


# In[194]:


data.age.hist()
plt.xlabel('age')
plt.ylabel('population')


# In[261]:


plt.bar(data.target,data.chol)
plt.xticks([0,1])
plt.xlabel('target')
plt.ylabel('Cholosterol')
plt.show()


# In[14]:


corr_matrix=data.corr()
corr_matrix['target'].sort_values(ascending=False)


# In[304]:


data[att].describe()


# In[15]:


from pandas.plotting import scatter_matrix


# In[363]:


fig,ax = plt.subplots(2,2,figsize=(10,8))
ax[0,0].hist(data[att[0]])
ax[0,0].set_title('target')
ax[0,0].set_xticks([0,1])

ax[0,1].hist(data[att[1]])
ax[0,1].set_title('exang (exercise induced angina)')
ax[0,1].set_xticks([0,1])


ax[1,0].hist(data[att[2]])
ax[1,0].set_title('cp (chest pain level)')
ax[1,0].set_xticks([0,1,2,3])

ax[1,1].hist(data[att[3]])
ax[1,1].set_title('oldpeak')

fig.show()


# In[358]:


att = ['target','exang','cp','thalach','oldpeak']
scatter_matrix(data[att],figsize=(10,7))
plt.show()


# ### Standard Scaler 

# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


s = StandardScaler()


# In[12]:


new_data=s.fit_transform(data)


# In[13]:


new_data[0,:]


# # 1) Random Forest Classifier:

# In[367]:


from sklearn.ensemble import RandomForestClassifier


# In[368]:


X=data.drop('target',axis=1)
y= data['target']


# In[369]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[370]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[371]:


y_pred= rf.predict(X_test)


# In[372]:


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score,f1_score


# In[373]:


from sklearn.metrics import confusion_matrix


# In[374]:


confusion_matrix(y_test, y_pred)


# In[375]:


print(precision_score(y_test, y_pred))
print(recall_sore(y_test, y_pred))


# In[376]:


f1_score(y_test, y_pred)


# In[377]:


accuracy_score(y_test, y_pred)


# In[378]:


from sklearn.metrics import roc_curve,auc
fpr,tpr,threshold = roc_curve(y_test,y_pred)
auc(fpr,tpr)


# In[379]:


plt.plot(fpr,tpr,marker='.')
plt.xlabel('false positive rate')
plt.ylabel('true postive rate')
plt.show()


# In[380]:


sol = []
sol.append({'model':'RandomForest','precision_score':precision_score(y_test, y_pred),'recall_score':recall_score(y_test, y_pred),'roc_auc score':auc(fpr,tpr)})


# # 2) Logistic Regression: 

# In[382]:


from sklearn.linear_model import LogisticRegression


# In[383]:


log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)


# In[384]:


y_pred=log_reg.predict(X_test)


# In[385]:


print(confusion_matrix(y_test,y_pred))
print('precision:', precision_score(y_test,y_pred))
print('recall:',recall_score(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred))


# In[386]:


print('accuracy_score:',accuracy_score(y_test,y_pred))


# In[387]:


fpr,tpr,threshold = roc_curve(y_test,y_pred)
print('AUC score:',auc(fpr,tpr))


# In[389]:


plt.plot(fpr,tpr,marker='.')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()


# In[390]:


sol.append({'model':'Logistic Regression','precision_score':precision_score(y_test, y_pred),'recall_score':recall_score(y_test, y_pred),'roc_auc score':auc(fpr,tpr)})


# # 3) Perceptron: 

# In[393]:


from sklearn.linear_model import Perceptron
perc = Perceptron(penalty='l2')
perc.fit(X_train,y_train)


# In[394]:


y_pred = perc.predict(X_test)


# In[395]:


print(confusion_matrix(y_test,y_pred))
print('precision:', precision_score(y_test,y_pred))
print('recall:',recall_score(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred))


# In[396]:


print('accuracy_score:',accuracy_score(y_test,y_pred))


# In[397]:


fpr,tpr,threshold = roc_curve(y_test,y_pred)
print('AUC score:',auc(fpr,tpr))


# In[398]:


plt.plot(fpr,tpr,marker='.')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()


# In[399]:


sol.append({'model':'Perceptron','precision_score':precision_score(y_test, y_pred),'recall_score':recall_score(y_test, y_pred),'roc_auc score':auc(fpr,tpr)})


# # 4) Decision Tree Classifier:

# In[400]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
y_pred = decision_tree.predict(X_test)


# print(confusion_matrix(y_test,y_pred))
# print('precision:', precision_score(y_test,y_pred))
# print('recall:',recall_score(y_test,y_pred))
# print('f1_score:',f1_score(y_test,y_pred))

# In[402]:


print('accuracy_score:',accuracy_score(y_test,y_pred))


# In[403]:


fpr,tpr,threshold = roc_curve(y_test,y_pred)
print('AUC score:',auc(fpr,tpr))


# In[404]:


plt.plot(fpr,tpr,marker='.')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()


# In[405]:


sol.append({'model':'Decision Tree Classifier','precision_score':precision_score(y_test, y_pred),'recall_score':recall_score(y_test, y_pred),'roc_auc score':auc(fpr,tpr)})


# # 5) SVM (RBF Kernel)

# In[406]:


from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# In[407]:


print('accuracy_score:',accuracy_score(y_test,y_pred))


# In[408]:


print(confusion_matrix(y_test,y_pred))
print('precision:', precision_score(y_test,y_pred))
print('recall:',recall_score(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred))


# In[409]:


fpr,tpr,threshold = roc_curve(y_test,y_pred)
print('AUC score:',auc(fpr,tpr))


# In[410]:


plt.plot(fpr,tpr,marker='.')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()


# In[411]:


sol.append({'model':'SVM (RBF Kernel)','precision_score':precision_score(y_test, y_pred),'recall_score':recall_score(y_test, y_pred),'roc_auc score':auc(fpr,tpr)})


# # 6) SVM ( Linear Kernel):

# In[412]:


clf = SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# In[413]:


print(confusion_matrix(y_test,y_pred))
print('precision:', precision_score(y_test,y_pred))
print('recall:',recall_score(y_test,y_pred))
print('f1_score:',f1_score(y_test,y_pred))


# In[414]:


print('accuracy_score:',accuracy_score(y_test,y_pred))


# In[415]:


fpr,tpr,threshold = roc_curve(y_test,y_pred)
print('AUC score:',auc(fpr,tpr))


# In[416]:


plt.plot(fpr,tpr,marker='.')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()


# In[417]:


sol.append({'model':'SVM (Linear Kernel)','precision_score':precision_score(y_test, y_pred),'recall_score':recall_score(y_test, y_pred),'roc_auc score':auc(fpr,tpr)})


# # END

# In[482]:


df={}
for i in range(len(sol)):
    df[i]=pd.Series(sol[i])


# In[512]:


for i in range(len(df)):
    solution=pd.concat(df,axis=1)


# In[520]:


solution


# # End

# In[ ]:




