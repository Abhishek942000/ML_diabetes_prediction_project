#!/usr/bin/env python
# coding: utf-8

# # DIABETES PREDICTION USING MACHINE LEARNING

# In[ ]:





# In[109]:


import pandas as pd


# In[110]:


db = pd.read_csv(r"C:\Users\Abhishek Sawant\OneDrive\Desktop\CAPSTONE PROJECT 2\Diabetes Prediction dataset cp2.csv")


# In[34]:


db.diabetes.value_counts()


# In[35]:


db.head()


# In[36]:


db.tail()


# In[37]:


db.info()


# In[38]:


db.shape


# In[39]:


db.isnull().sum()


# In[41]:


# Here we see the description of numerical features. These might give us ideas about our future works.


# In[42]:


db.columns


# In[12]:


# There are total 9 columns. Out of those 9 columns there are 2 categorical column and 7 numberic columns.


# In[13]:


db['gender'].value_counts()


# In[14]:


db['hypertension'].value_counts()


# In[15]:


db['smoking_history'].value_counts()


# In[15]:


# Since there is no missing values or any type of outliers I am leaving the data as it is. 
# We can change the age into groups but I prefer not to.


# # EDA

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns


# # UNIVARIATE ANALYSIS

# In[17]:


sns.countplot(x=db['gender'])


# In[18]:


# There is almost 18% difference between Females and Males. 
# Also there is gender title as Other, I am not removing it because it can be a Transgender data.


# In[19]:


sns.histplot(x=db['age'], bins=8)


# In[20]:


# People have age between 40-60 are high and others are evenly distributed.


# In[21]:


sns.countplot(x=db['hypertension'])
plt.xticks(ticks=[0,1],labels=['No', 'Yes'])
plt.xlabel('Hypertension')


# In[22]:


# There are almost 90,000 people without Hypertension.


# In[23]:


sns.countplot(x=db['heart_disease'])
plt.xticks(ticks=[0,1],labels=['No', 'Yes'])
plt.xlabel('Heart Disease')


# In[24]:


# Almost 95,000 people don't have any kind of heart disease.


# In[25]:


sns.countplot(x=db['smoking_history'])
plt.xlabel('Smoking History')


# In[26]:


# There is 'No Info' about almost 36000 people whereas 35000 people have never done smoking.
# currently - 9000
# Former Smokers - 9000
# Ever - 4000
# Not Currently - 7000


# In[27]:


sns.histplot(x=db['bmi'], bins=8)
plt.xlabel('BMI')


# In[28]:


# Maximum number of people have BMI of 30.


# In[29]:


sns.histplot(x=db['HbA1c_level'], bins=9)
plt.xlabel('HbA1c Level')


# In[30]:


# Maximum number of people have HbA1c level between 6-6.5.


# In[31]:


sns.histplot(x=db['blood_glucose_level'], bins=6)
plt.xlabel('Blood Glucose Level')


# In[32]:


# Averagely people have 100-200 Blood Glucose Level whereas some have 200-300.


# In[33]:


sns.countplot(x=db['diabetes'])
plt.xticks(ticks=[0,1], labels=['No', 'Yes'])
plt.xlabel('Diabetes')


# In[34]:


# There are almost 90000 people who dont have diabetes


# # BIVARIATE ANALYSIS

# In[36]:


sns.boxplot(x=db['diabetes'], y=db['age'])
plt.xticks(ticks=[0,1], labels=['No', 'Yes'])
plt.xlabel('Diabetes')


# In[37]:


# Average age of people who have diabetes is 60 and who don't have is 40. 
# This shows as the people age goes above 50 tend to get Diabetes.


# In[38]:


sns.scatterplot(x=db['age'], y=db['bmi'], alpha=0.5)


# In[39]:


# Above plot showing a slight connection between age and bmi which is as 
# the age goes till 30-50 bmi grows and after it start decreasing. 
# The increase is high but the decrease is slow.


# In[40]:


sns.boxplot(x=db['hypertension'], y=db['age'], palette='viridis')
plt.xticks(ticks=[0,1], labels=['No', 'Yes'])
plt.xlabel('Hypertension')


# In[41]:


# Average age of people who have hypertension is 60 and who don't have is 40. 
# This shows as the people age goes above 50 people start having hypertension issue.


# In[42]:


plt.figure(figsize=(10,4))
sns.boxplot(y=db['age'], x=db['blood_glucose_level'], palette='viridis')


# In[43]:


# Average glucose level in blood is between 80-200 in the people with age group 20-50. 
# but people with more than 50 years have glucose level from 220-300.


# In[44]:


plt.figure(figsize=(10,4))
sns.boxplot(x=db['HbA1c_level'], y=db['age'], palette='viridis')


# In[45]:


# The average age of people with HbA1c level 3.5-6.6 is 40, but it rises from 6.8-9.0 in older people whose average age is 60.


# In[46]:


import warnings
warnings.filterwarnings('ignore')


# In[47]:


sns.barplot(x=db['smoking_history'], y=db['hypertension'], hue=db['gender'], palette='viridis', ci=None)


# In[48]:


# Former Smokers are facing high hypertension in regard of all other peoples who have smoking history. 
# And males have higher hypertension than females.


# In[49]:


sns.barplot(x=db['smoking_history'], y=db['heart_disease'], hue=db['gender'], palette='viridis', ci=None)


# In[50]:


# People who are former and regular smokers have high rate of heart disease 
# and out of those males have very high chance of getting heart disease.


# In[51]:


sns.boxplot(x=db['smoking_history'], y=db['blood_glucose_level'], palette='viridis')


# In[52]:


# There is not much difference in glucose level of with respect of smoking history.


# In[55]:


plt.figure(figsize=(12,5))
sns.boxplot(x=db['smoking_history'], y=db['age'], palette='coolwarm')


# In[56]:


# Mostly people under age of 30 have never tried smoking. And people after 60 try to leave smoking.
# Understanding the age distribution across different smoking histories can help tailor health monitoring and 
# interventions more effectively.


# In[49]:


plt.figure(figsize=(10,6))
sns.heatmap(db.corr(), annot=True)


# In[50]:


#As we can see on the heat map, HbA1c Level and Blood Glucose Level is correlated with having diabetes. 
#This might help us track a pattern of our data distribution. 
#In fact, every other feature is correlated with diabetes, but the correlation level is not high.


# # MODEL BUILDING

# # LOGISTIC REGRESSION

# In[134]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

db['gender'] = le.fit_transform(db['gender'])
db['smoking_history'] = le.fit_transform(db['smoking_history'])

db.head()


# In[135]:


db1=db


# In[136]:


from sklearn.model_selection import train_test_split


# In[137]:


train_db , test_db = train_test_split(db, test_size = .2)


# In[138]:


db.head()


# In[139]:


c1=train_db[train_db.diabetes==1]


# In[140]:


c1.shape


# In[141]:


train_db=pd.concat([train_db,c1,c1,c1,c1,c1,c1,c1,c1,c1],axis=0)


# In[190]:


train_db_x = train_db.iloc[:  , 0: -1] # all x
train_db_y = train_db.iloc[:, -1] # only y

test_db_x = test_db.iloc[:  , 0: -1] # all x from test
test_db_y = test_db.iloc[:, -1]


# In[191]:


train_db['diabetes'].value_counts()


# In[192]:


test_db_x.head()


# In[193]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()


# In[194]:


logreg.fit(train_db_x,train_db_y)


# In[195]:


pred_test = logreg.predict(test_db_x)


# In[196]:


from sklearn.metrics import confusion_matrix,classification_report


# In[197]:


confusion_matrix(test_db_y, pred_test)


# In[198]:


print(classification_report(test_db_y, pred_test))


# # DECISION TREE

# In[199]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(class_weight='balanced')
dt.fit(train_db_x,train_db_y)


# In[200]:


predict_test_dt =dt.predict(test_db_x)
predict_test_dt


# In[201]:


from sklearn.metrics import confusion_matrix ,accuracy_score, recall_score , precision_score ,f1_score,classification_report

confusion_matrix(test_db_y,predict_test_dt)


# In[202]:


print(classification_report(test_db_y,predict_test_dt))


# In[ ]:





# # RANDOM FOREST

# In[217]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( class_weight='balanced')
rfc.fit(train_db_x,train_db_y)


# In[218]:


pred_test_rfc = rfc.predict(test_db_x)


# In[219]:


from sklearn.metrics import confusion_matrix


# In[220]:


tab_rfc = confusion_matrix(test_db_y, pred_test_rfc)
tab_rfc


# In[221]:


print(classification_report(test_db_y, pred_test_rfc))


# # K-Nearest Neighbors (KNN)

# In[142]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

db['gender'] = le.fit_transform(db['gender'])
db['smoking_history'] = le.fit_transform(db['smoking_history'])

db.head()


# In[143]:


from sklearn.model_selection import train_test_split
train_db,test_db= train_test_split(db, test_size=.2)


# In[144]:


train_db_x= train_db.iloc[:,0:-1]
train_db_y= train_db.iloc[:,-1]

test_db_x= test_db.iloc[:,0:-1]
test_db_y =test_db.iloc[:,-1]


# In[145]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(weights='uniform')


# In[146]:


knn.fit(train_db_x,train_db_y)


# In[147]:


pred_knn=knn.predict(test_db_x)


# In[148]:


from sklearn.metrics import confusion_matrix, classification_report


# In[149]:


confusion_matrix(test_db_y,pred_knn)


# In[150]:


print(classification_report(test_db_y,pred_knn))


# # SUPPORT VECTOR MACHINE

# In[207]:


from sklearn.svm import SVC


# In[208]:


svc_db = SVC(kernel='linear')
# different kernels in svm


# In[209]:


svc_db.fit(train_db_x, train_db_y)


# In[210]:


pred_svc = svc_db.predict(test_db_x)


# In[211]:


from sklearn.metrics import confusion_matrix , classification_report


# In[212]:


confusion_matrix(test_db_y, pred_svc)


# In[213]:


print(classification_report(test_db_y, pred_svc))


# # GRADIENT BOOSTING

# In[214]:


from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(train_db_x,train_db_y)


# In[215]:


pred_test_gb = model.predict(test_db_x)

tab_gb = confusion_matrix(test_db_y, pred_test_gb)
tab_gb


# In[216]:


print(classification_report(test_db_y, pred_test_gb))


# In[ ]:




