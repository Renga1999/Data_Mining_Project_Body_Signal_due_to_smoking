#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot


# In[2]:


df = pd.read_csv("./smoking.csv")


# In[3]:


# get rid of ID column
df = df.drop("ID", axis = 1)
df.head(5)


# In[7]:


df.info()


# In[4]:


# NULL VALUES

df.isnull().sum().sum()

## there are no null values in our dataset at all!


# ## EDA - Cols 0-6

# #### Visual Representation of Categorical Variables

# In[9]:


colors = ("#f2b31f", "#d59bf2")
explode = (0.0, 0.15)
labels = ["Males", "Females"]
#fig, axes = plt.subplots(1, 2, figsize = (11, 7))
#axes = axes.flatten()

# sns.barplot(x = df["gender"].value_counts().index,
#             y = df["gender"].value_counts(),
#             data = df, palette = "Spectral", saturation = 2).set(title = "Barplot showing Frequency of Gender types")

plt.figure(figsize = [7, 7], clear = False , facecolor = '#e6f0e7')
df["gender"].value_counts().plot.pie(colors=colors, labels=labels, 
                                    explode=explode, autopct = '%1.0f%%').set_title("Pie Chart showing Frequency of Gender types");


# From this plot, we see the distribution of the entire data set categorized by genders.
# 64% of our data set represents males, while 36% represents females.

# In[6]:


explode = (0, 0.1)
colors = ["#e62592", "#25e6a5"]
labels =['Non smokers','Smokers']
# sns.barplot(x=df['smoking'].value_counts().index,
#            y = df['smoking'].value_counts(),
#            data = df, palette = 'husl', saturation = 5).set(title = "Barplot showing Frequency of Smoking Variable")
# plt.show()


plt.figure(figsize = [7, 7], clear = False , facecolor = '#e6f0e7')
df["smoking"].value_counts().plot.pie(explode=explode, labels=labels, colors=colors, autopct = '%1.0f%%').set_title("Pie Chart showing Frequency of Smoking Variable");


# This plot shows us the dataset grouped into Smokers and Non-Smokers. The Smokers make up about 63% of the entire dataset, and non-smokers account for 37%.

# In[7]:


sns.countplot(data=df,x='smoking',hue='gender', palette = "husl").set(title = "Barplot showing Smokers and Non-Smokers, grouped by Gender")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# From the plot above, we can see that men make up the majority of the smokers group, by far. The non-smoking group comprises of mostly women!

# #### Visual Representation of Numerical Variables

# In[20]:


#Plot histograms for 

num_list = list(df.columns[1:7])

fig = plt.figure(figsize=(10,45))

for i in range(len(num_list)):
    plt.subplot(15,2,i+1)
    plt.title(num_list[i])
    plt.hist(df[num_list[i]],color='#85a832',alpha=0.5)

plt.tight_layout()


# The numerical variables: age, height, weight and waist are normally distributed for the most part. The latter 3 show some skew (especially weight) but the normal pattern can still be seen to an extent. For eyesight (left and right), the possible values range from 0 to 2, hence the histograms taking that shape. VAlues between and 0 and 1 have the highest frequencies.

# In[22]:


for w in num_list:
    if w in num_list:
        sns.distplot(df[w], kde=True)
        plt.show()


# In[13]:


df['weight(kg)'].unique()


# In[13]:


numerical = df.columns[1:7]
numerical


# In[14]:


plt.figure(figsize= (6,6))
sns.heatmap(df[numerical].corr(), annot=True)
plt.show()
plt.clf()


# A correlation matrix showing the relationships among all the numerical variables. Height, weight and waist are the most positively correlated variables, just as we would expect. Height plays a role in influencing weight, which in turn is a determinant for the circumference of ones waist.

# #### Some group by plots

# ##### Age Distribution grouped by Gender and Smoking

# In[9]:


sns.catplot(x = "gender",
            y = "age",
            kind = "box",
            hue = "smoking",
            color = '#2b4532',
            data = df, saturation = 4, height = 4, aspect = 1.5,
            margin_titles = True).set(title = "Age by Gender and Smoking");


# In[28]:


sns.catplot(x = "gender",
            y = "height(cm)",
            kind = "box",
            hue = "smoking",
            color = '#54ccb0',
            data = df, saturation = 4, height = 5, aspect = 1.5,
            margin_titles = True).set(title = "Height (cm) by Gender and Smoking");


# In[30]:


sns.catplot(x = "gender",
            y = "weight(kg)",
            kind = "box",
            hue = "smoking",
            color = '#5eb5e0',
            data = df, saturation = 4, height = 5, aspect = 1.5,
            margin_titles = True).set(title = "Weight (kg) by Gender and Smoking");


# In[16]:


sns.catplot(x = "height(cm)",
            y = "hemoglobin(kg)",
            kind = "hist",
            hue = "smoking",
            color = '#5eb5e0',
            data = df, saturation = 4, height = 5, aspect = 1.5,
            margin_titles = True).set(title = "Weight (kg) by Gender and Smoking");


sns.histplot(data=df, x='height(cm)', y='hemoglobin')


# In[31]:


sns.catplot(x = "gender",
            y = "waist(cm)",
            kind = "box",
            hue = "smoking",
            color = '#bd77e6',
            data = df, saturation = 4, height = 5, aspect = 1.5,
            margin_titles = True).set(title = "Waist (cm) by Gender and Smoking");


# In[32]:


sns.catplot(x = "gender",
            y = "eyesight(left)",
            kind = "box",
            hue = "smoking",
            color = '#bd77e6',
            data = df, saturation = 4, height = 5, aspect = 1.5,
            margin_titles = True).set(title = "Waist (cm) by Gender and Smoking");


# In[36]:


sns.catplot(x = "gender",
            y = "eyesight(right)",
            kind = "box",
            hue = "smoking",
            color = '#bd77e6',
            data = df, saturation = 4, height = 7.5, aspect = 1.5,
            margin_titles = True).set(title = "Waist (cm) by Gender and Smoking");


# Data Balancing

# In[3]:


get_ipython().system('pip install imblearn')


# In[4]:


from imblearn.over_sampling import SMOTE


# In[5]:


from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from collections import Counter


# In[1]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train.value_counts()


# In[ ]:




