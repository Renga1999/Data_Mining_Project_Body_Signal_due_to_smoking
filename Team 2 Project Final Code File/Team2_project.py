# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics

# %%
# reading the source file 
df = pd.read_csv("E:\DataMining Project\Data_Mining_Project_Body_Signal_due_to_smoking\smoking.csv")
df.head()

# %% [markdown]
# # Data Dictionary 
# 
# - ID : index
# - gender
# - age : 5-years gap
# - height(cm)
# - weight(kg)
# - waist(cm) : Waist circumference length
# - eyesight(left)
# - eyesight(right)
# - hearing(left)
# - hearing(right)
# - systolic : Blood pressure
# - relaxation : Blood pressure
# - fasting blood sugar
# - Cholesterol : total
# - triglyceride
# - HDL : cholesterol type
# - LDL : cholesterol type
# - hemoglobin
# - Urine protein
# - serum creatinine
# - AST : glutamic oxaloacetic transaminase type
# - ALT : glutamic oxaloacetic transaminase type
# - Gtp : γ-GTP
# - oral : Oral Examination status
# - dental caries
# - tartar : tartar status
# - smoking : Target variable 

# %%
# size of the dataset
df.shape

# %%
# checking the types of the variables 
df.info()

# %%
# a quick overlook on how the data 
df.describe()

# %%
# checking for NULL's
df.isnull().sum()

# %%
# plotting the correlation to see which variables are strongly correlated 
sns.heatmap(df.corr().round(3), annot= True).figure.set_size_inches(20, 10)
plt.title(label="Correlation plot for the dataset")

# %% [markdown]
# # Exploratory Data Analysis 

# %%
# checking for data imbalance 

sns.countplot(df, x="smoking").set(title="Distribution of the target variable")

# %%
# checking the distribution of hearing(left)

sns.displot(df["hearing(left)"])

# %%
# checking the distribution of hearing(right)

sns.displot(df["hearing(right)"])

# %%
# checking how many unique values are on hearing left 

df["hearing(left)"].unique()

# %%
# Distribution of Cholesterol

sns.displot(df["Cholesterol"])

# %%
# Distribution of systolic

sns.displot(df["systolic"])

# %%
# Distribution of relaxation

sns.displot(df["relaxation"])

# %%
# Distribution of triglyceride

sns.displot(df["triglyceride"])

# %%
# checking the relationship between triglyceride, fasting blood sugar by gender

sns.scatterplot(data = df, y = "triglyceride", x= "fasting blood sugar",hue = "gender",size= "smoking")

# %%
# checking the relationship between Cholesterol, fasting blood sugar by gender

sns.scatterplot(data = df, y = "Cholesterol", x= "fasting blood sugar",hue = "gender")

# %%
# boxplot for how cholestrol is for different genders based on their smoking status 

sns.boxplot(data = df, x ="gender" , y= "Cholesterol", hue="smoking")

# %%
# checking the relationship between fasting blood sugar by genders based on their smoking status

sns.boxplot(data = df, x ="smoking" , y= "fasting blood sugar",hue="gender")

# %%
# checking the relationship for systolic by genders based on their smoking status

sns.boxplot(data = df, x ="smoking" , y= "systolic",hue="gender")

# %%
# checking the relationship between smoking and ALT 

sns.boxplot(data = df, x ="smoking" , y= "ALT")

# %%
# Relationship between smoking and relaxation

sns.boxplot(data = df, x ="gender" , y= "relaxation",hue="smoking").set(title = "Relationship between smoking and relaxation")

# %%
# checking the relationship for triglyceride by genders based on their smoking status

sns.boxplot(data = df, x ="smoking" , y= "triglyceride",hue="gender")

# %%
# checking the relationship for CHolesterol and hearing based on gender 

sns.barplot(data = df, y = "Cholesterol", x= "hearing(left)",hue = "gender")

# %%
# checking the relationship for hearing based on their smoking status

sns.barplot(data = df, y = "hearing(right)", x= "smoking")

# %%
# checking the relationship for age and height based on gender

sns.boxplot(data = df, x = "age",y ="height(cm)",hue="gender")

# %%
#subsetting the data as males and females 

data_male = df[df["gender"]=="M"]
data_female = df[df["gender"]=="F"]


# %%
# scatter plot to see HDL and triglyceride for males

sns.scatterplot(data = data_male, x = "HDL",y ="triglyceride",hue="gender")

# Your healthcare provider classifies high triglyceride levels as:

# Mild: 150-199 mg/dL.
# Moderate: 200-499 mg/dL.
# Severe: Greater than 500 mg/dL.
# What are risk factors for high triglycerides?
# Factors that may raise triglyceride levels include:

# Excessive alcohol use.
# Family history of high cholesterol.
# Liver disease or kidney disease.
# Medications, including diuretics, hormones, corticosteroids and beta blockers.
# Menopause.
# Obesity.
# Smoking.
# Thyroid disease.
# Unregulated diabetes.
# A diet high in sugar and simple carbohydrates.

# %%
# scatter plot to see HDL and triglyceride for females
sns.scatterplot(data = data_female, x = "HDL",y ="triglyceride",hue="gender")

# %%
# males tend to smoke in their early stage of life i.e around 20 years whereas women generally tend to start smoking most likely at the age of 30

sns.boxplot(data= df , x ="gender",y="age",hue= "smoking").set(title = "Relationship between age and smoking")

# %%
# Relationship between triglyceride and smoking based on gender

sns.boxplot(data= df , x ="gender",y="triglyceride",hue= "smoking").set(title = "Relationship between triglyceride and smoking based on gender")

# %%
# Relationship between weight and smoking
sns.boxplot(data= df , x ="gender",y="weight(kg)",hue= "smoking").set(title = "Relationship between weight and smoking")
# we can say that weight is not a major concern if a person smokes
# generally we have a misconception that people who smoke tend to weigh less, but the obvservation does not prove our assumption.

# %%
# Relationship between gender, height and smoking 

sns.boxplot(data= df , x ="gender",y="height(cm)",hue= "smoking")

# %%
# Looking at the distribution of the target variable  

df['smoking'].value_counts(normalize = True)


# %%
# Distribution of the Target variable before balancing

sns.countplot(data=df,x='smoking', palette = "husl").set(title = "Distribution of the Target variable before balancing")
plt.legend()
plt.show()

# %%
from imblearn.over_sampling import SMOTENC 
from sklearn.model_selection import train_test_split
from collections import Counter

# %%
# splitting the data into X and y

X = df.iloc[:,0:26]
X
y= df["smoking"]
y = pd.DataFrame(y,columns = ["smoking"])
y

# %%
# using train_test_split to split the data into X_train, X_test, y_train, y_test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train.value_counts()

# %% [markdown]
# # Start of the balancing techniques

# %%
# looking at the datatypes in X_train
X_train.dtypes

# %%
# balancing the data out using SMOTE Technique 

sm = SMOTENC(random_state=42,categorical_features=[1,23,25])
X_res, y_res = sm.fit_resample(X_train, y_train)


# %%
# percentage of values after balancing

y_res.value_counts(normalize = True)

# %%
# Looking at the size of the balanced train data 

X_res.shape

# %%
# plot for target variable after balancing 

sns.countplot(data=y_res,x='smoking', palette = "husl").set(title = "Distribution of the Target variable after balancing")
plt.legend()
plt.show()

# %% [markdown]
# # Modeling 

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score


# %%
# getting dummies for train 

X_res = pd.get_dummies(X_res)
X_res
X_res = X_res.iloc[:,1:26]


# %%
# getting dummies for test

X_test = pd.get_dummies(X_test)
X_test = X_test.iloc[:,1:26]

# %%
# copying the balanced data to a new variable named X_train and y_train , and leaving the X_test and Y_test alone

X_train = X_res
y_train = y_res

# %%
# looking if the X_train and y_train have the same datapoints

print(X_train.shape)
print(y_train.shape)


# %% [markdown]
# Logistic Regression 

# %%
# building a Logistic Regression model

classifier = LogisticRegression()
model = classifier.fit(X_train,y_train)

# Predicting the X_test based on the logistic model fit

y_pred = model.predict(X_test)

# Creating a confusion matrix for evaluation 
confusion = confusion_matrix(y_test, y_pred)

# %%
# printing the confusion matrix

disp = ConfusionMatrixDisplay(confusion)
disp.plot()

# %%
# printing the classification report 

print(classification_report(y_test, y_pred))

# %%
# printing the AUC-ROC curve 

disp = RocCurveDisplay.from_estimator(model, X_test, y_test, alpha=0.8)
plt.show()

# %% [markdown]
# Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# %%
# getting the feature names 

feature_names = X_train.columns
feature_names

# %%
# building a decision tree classsifer and then fitting it to the training data

clf1 = DecisionTreeClassifier( random_state = 42)
model2  = clf1.fit(X_train, y_train)

# %%
# prediction on the test data 

y_Pred = model2.predict(X_test)

# %%
# printing the confusion matrix

confusion = confusion_matrix(y_test, y_Pred)
disp = ConfusionMatrixDisplay(confusion)
disp.plot()

# %%
# printing the classification report for the model 

print(classification_report(y_test, y_Pred))

# %%
# printing the AUC-ROC curve for decision tree

disp = RocCurveDisplay.from_estimator(model2, X_test, y_test, alpha=0.8)
plt.show()

# %% [markdown]
# Random Forest Classifier 

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
# BUilding a Random Forest Classifier and fitting it to the training data 

clf2=RandomForestClassifier(random_state= 42,min_samples_split=2,n_estimators = 500)
model3 =clf2.fit(X_train, y_train)

# %%
# prediction based on the random forest classifier for the test set

y_Predict = model3.predict(X_test)

# %%
# printing the confusion matrix

confusion = confusion_matrix(y_test, y_Predict)
disp = ConfusionMatrixDisplay(confusion)
disp.plot()

# %%
# Printing the Classification report for the test set

print(classification_report(y_test, y_Predict))

# %%
# obtainind the feature importance from the random forest model 

feature_importances = pd.DataFrame(model3.feature_importances_, index =X_train.columns,  columns=['importance']).sort_values('importance', ascending=False)

# %%
# printing the important features

print(feature_importances)

# %%
# plotting the feature importance plot from random forest classifer 

feature_importances.plot(kind="bar").set(title="Feature Importance Plot")

# %%
# Plotting the AUC-ROC curve for Random Forest 

disp = RocCurveDisplay.from_estimator(model3, X_test, y_test, alpha=0.8)

plt.show()

# %% [markdown]
# Random forest with feature selection

# %%
# subsetting the data based on the feature selected variables from the previous random forest model 

X_train1 = X_train[["gender_F","gender_M","Gtp","hemoglobin","height(cm)","triglyceride","serum creatinine","age","waist(cm)","ALT","LDL","HDL"]]
X_test1 = X_test[["gender_F","gender_M","Gtp","hemoglobin","height(cm)","triglyceride","serum creatinine","age","waist(cm)","ALT","LDL","HDL"]]

# %%
# looking at the shape of the testing data 

X_test1.shape

# %%
# Building a random forest classifier for the feature selected variables 

clf3=RandomForestClassifier(random_state= 42,min_samples_split=2,n_estimators = 500)
model4 =clf3.fit(X_train1, y_train)


# %%
# prediction on the test set based on the random forest classifier 

y_Predict1 = model4.predict(X_test1)

# %%
# printing the confusion matrix

confusion = confusion_matrix(y_test, y_Predict1)
disp = ConfusionMatrixDisplay(confusion)
disp.plot()

# %%
# Printing the classification report for the testing set

print(classification_report(y_test, y_Predict1))

# %%
# Printing the AUC-ROC curve for the random forest model

disp = RocCurveDisplay.from_estimator(model4, X_test1, y_test, alpha=0.8)
plt.show()

# %% [markdown]
# Building the Random forest model after scaling the values

# %%
# Looking at the numerical data to scale those numerical data 

print(X_train.select_dtypes(exclude = object).columns)
print(X_test.select_dtypes(exclude = object).columns)

# %%
# initializing the min max scaler and then fitting the data for train and test 

mm_scaler = MinMaxScaler()
scaling_columns = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
       'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic',
       'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
       'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 'AST',
       'ALT', 'Gtp', 'dental caries', 'gender_F', 'gender_M', 'oral_Y']
mm_scaler.fit(X_train[scaling_columns])
X_train[scaling_columns] = mm_scaler.transform(X_train[scaling_columns])
X_test[scaling_columns] = mm_scaler.transform(X_test[scaling_columns])

# %%
# Building the random forest classifier and then fitting the training data

rf = RandomForestClassifier(random_state =42)

rf.fit(X_train,y_train)

# predicting the probability on train and test set
pred_tr = rf.predict_proba(X_train)[:,1]
pred_te = rf.predict_proba(X_test)[:,1]

# prediciting the actual values on the test set
y_Predict2 = rf.predict(X_test)

# printing the auc score for train and test
print(roc_auc_score(y_train,pred_tr))
print(roc_auc_score(y_test,pred_te))

# %%
# printing the confusion matrix

confusion = confusion_matrix(y_test, y_Predict2)
disp = ConfusionMatrixDisplay(confusion)
disp.plot()

# %%
# printing the classification report for the random forest classifier with scaled features

print(classification_report(y_test, y_Predict2))

# %%
# printing the AUC-ROC curve for random forest model with scaled features 

rfc_disp = RocCurveDisplay.from_estimator(rf, X_test, y_test, alpha=0.8)
plt.show()

# %% [markdown]
# KNN 

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
# Initializing a random value for k ans looking at the model performance 

k = 5  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)
print("Accuracy of model at K=5 is",accuracy_score(y_test, Pred_y))

# %%
# running the models for different k values and storing the accuracy in acc variable to plot K vs Accuracy plot

acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))

# %%
# plotting the Accuracy vs K value for KNN

plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

# %%
# choosing k value as 2 based on the plot above 

neigh = KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train)
yhat1 = neigh.predict(X_test)

# %%
# printing the confusion matrix

confusion = confusion_matrix(y_test, yhat1)
disp = ConfusionMatrixDisplay(confusion)
disp.plot()

# %%
# printing the classification report for test set

print(classification_report(y_test, yhat1))

# %%
# plotting the AUC-ROC curve for KNN model

disp = RocCurveDisplay.from_estimator(neigh, X_test, y_test, alpha=0.8)

plt.show()

# %% [markdown]
# SVM classifier

# %%
from sklearn import svm

# %%
# building an SVM classifier with linear kernel and making prediction based on the test set

clf = svm.SVC(kernel='linear')
model5 = clf.fit(X_train,y_train)
y_pred = model5.predict(X_test)

# %%
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))

# F1 Score 
print("F1_score:",metrics.f1_score(y_test, y_pred))

# %%
# printing the classification report for SVM classifier 

print(classification_report(y_test, y_pred))

# %%



