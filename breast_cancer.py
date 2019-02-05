# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:20:34 2018

@author: Shriya Prasad
"""

import pandas as pd
from sklearn import metrics
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.naive_bayes as NB
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

##############################################################################################################
#read csv file
df = pd.read_csv("breast_cancer.csv")
#check if the file has entries
print(df.head())
'''
#check if there are any NaN
print("Checking if the values are null") 
print(df.isnull().any())
'''
#Replace values M = 0 , B = 1; Malignant and benign
df['diagnosis'].replace(['M', 'B'], [1, 0], inplace = True)
#print(df['diagnosis'])

#explore the data 
print("shape of dataset:\n", df.shape) #number of rows and columns
#print("Type of the data:\n",df.dtypes) #data tye of each of the columns
#df = df.iloc[1:,1:32]

print("Statistics of the data\n") #statistics of each column
for i in range (2,32): #not considering the ID column and the diagnosis
    print(df.columns[i])
    print(df.iloc[i].describe())
    print("\n")
'''
##############################################################################################################3
#plot bar plots for each columns

#diagnosis of patients
diag = df['diagnosis']
diag1 = (diag == 0).sum()
diag2 = (diag == 1).sum()
a=[diag1,diag2]
print(a)
b=['Benign','Malignant']
df2 = pd.DataFrame(a, index = b)
ax = df2.plot(kind='bar', legend = False, width = .5, rot = 0,color = "plum", figsize = (6,5))
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%d' % int(height),ha='center', va='bottom')
plt.ylim(0, 1000)
plt.xlabel('Diagnosis')
plt.ylabel('All cases')
plt.title('Diagnosis of patients')
plt.show()
  
######################################################################################################################
#plot historgram for each columns

print("Features for mean")
features_mean= list(df.columns[2:12])
bins = 12
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    
    plt.subplot(rows, 2, i+1)
    
    sns.distplot(df[df['diagnosis']==1][feature], bins=bins, color='red', label='M');
    sns.distplot(df[df['diagnosis']==0][feature], bins=bins, color='blue', label='B');
    
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

#histogram - se
print("Features for SE")
features_se= list(df.columns[12:22])
bins = 12
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_se):
    rows = int(len(features_se)/2)
    
    plt.subplot(rows, 2, i+1)
    
    sns.distplot(df[df['diagnosis']==1][feature], bins=bins, color='red', label='M');
    sns.distplot(df[df['diagnosis']==0][feature], bins=bins, color='blue', label='B');
    
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

#histogram - worst
print("Features for worst")
features= list(df.columns[22:32])
bins = 12
plt.figure(figsize=(15,15))
for i, feature in enumerate(features):
    rows = int(len(features)/2)
    
    plt.subplot(rows, 2, i+1)
    
    sns.distplot(df[df['diagnosis']==1][feature], bins=bins, color='red', label='M');
    sns.distplot(df[df['diagnosis']==0][feature], bins=bins, color='blue', label='B');
    
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
'''
#####################################################################################################################################

#Heatmap
df1 = df.iloc[1:,1:32]
corr = df1.corr()
#corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')
plt.show()
#######################################################################################################333
'''
#boxplot
print("Boxplot of the data")
features_mean= list(df.columns[2:12])
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    plt.subplot(rows,2, i+1)
    sns.boxplot(x='diagnosis', y=feature, data=df, palette="Set1")

plt.tight_layout()
plt.show()

features_se= list(df.columns[12:22])
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_se):
    rows = int(len(features_mean)/2)
    plt.subplot(rows,2, i+1)
    sns.boxplot(x='diagnosis', y=feature, data=df, palette="Set1")

plt.tight_layout()
plt.show()

features_worst= list(df.columns[22:32])
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_worst):
    rows = int(len(features_mean)/2)
    plt.subplot(rows,2, i+1)
    sns.boxplot(x='diagnosis', y=feature, data=df, palette="Set1")

plt.tight_layout()
plt.show()


######################################################################################################

#correlation between diagnosis and radius_mean
diag = df1['diagnosis']
corr_radius = df1['radius_mean'].corr(diag)
print("correlation between diagnosis and radius_mean",corr_radius)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_texture_mean = df1['texture_mean'].corr(diag)
print("correlation between diagnosis and texture_mean:",corr_texture_mean)

#correlation between diagnosis and area_mean
diag = df1['diagnosis']
corr_area_mean = df1['area_mean'].corr(diag)
print("correlation between diagnosis and area_mean:",corr_area_mean)

#correlation between diagnosis and perimeter_mean
diag = df1['diagnosis']
corr_perimeter_mean = df1['perimeter_mean'].corr(diag)
print("correlation between diagnosis and perimeter_mean:",corr_perimeter_mean)

#correlation between diagnosis and smoothness_mean
diag = df1['diagnosis']
corr_smoothness_mean = df1['smoothness_mean'].corr(diag)
print("correlation between diagnosis and smoothness_mean:",corr_smoothness_mean)

#correlation between diagnosis and compactness_mean
diag = df1['diagnosis']
corr_compactness_mean = df['compactness_mean'].corr(diag)
print("correlation between diagnosis and compactness_mean:",corr_compactness_mean)

#correlation between diagnosis and concavity_mean
diag = df1['diagnosis']
corr_concavity_mean = df1['concavity_mean'].corr(diag)
print("correlation between diagnosis and concavity_mean:",corr_concavity_mean)

#correlation between diagnosis and concave points_mean
diag = df1['diagnosis']
corr_concavepoints_mean = df['concave points_mean'].corr(diag)
print("correlation between diagnosis and concave points_mean:",corr_concavepoints_mean)

#correlation between diagnosis and symmetry_mean
diag = df1['diagnosis']
corr_symmetry_mean = df['symmetry_mean'].corr(diag)
print("correlation between diagnosis and symmetry_mean:",corr_symmetry_mean)

#correlation between diagnosis and fractal_dimension_mean
diag = df1['diagnosis']
corr_fd_mean = df['fractal_dimension_mean'].corr(diag)
print("correlation between diagnosis and fractal_dimension_mean:",corr_fd_mean)

#correlation between diagnosis and radius_se
diag = df1['diagnosis']
corr_radius_se = df['radius_se'].corr(diag)
print("correlation between diagnosis and radius_se:",corr_radius_se)

#correlation between diagnosis and texture_se
diag = df1['diagnosis']
corr_texture_se = df['texture_se'].corr(diag)
print("correlation between diagnosis and texture_se:",corr_texture_se)

#correlation between diagnosis and perimeter_se
diag = df1['diagnosis']
corr_perimeter_se = df['perimeter_se'].corr(diag)
print("correlation between diagnosis and perimeter_se:",corr_perimeter_se)

#correlation between diagnosis and area_se
diag = df1['diagnosis']
corr_area_se = df['area_se'].corr(diag)
print("correlation between diagnosis and area_se:",corr_area_se)

#correlation between diagnosis and smoothness_se
diag = df1['diagnosis']
corr_smoothness_se = df['smoothness_se'].corr(diag)
print("correlation between diagnosis and smoothness_se:",corr_smoothness_se)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_compactness_se = df['compactness_se'].corr(diag)
print("correlation between diagnosis and compactness_se:",corr_compactness_se)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_concavity_se = df['concavity_se'].corr(diag)
print("correlation between diagnosis and concavity_se:",corr_concavity_se)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_cp_se = df['concave points_se'].corr(diag)
print("correlation between diagnosis and Concave Points_se:",corr_cp_se)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_symmetry_se = df['symmetry_se'].corr(diag)
print("correlation between diagnosis and symmetry_se:",corr_symmetry_se)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_fd_se = df['fractal_dimension_se'].corr(diag)
print("correlation between diagnosis and fractal_dimension_se:",corr_fd_se)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_radius_worst = df['radius_worst'].corr(diag)
print("correlation between diagnosis and radius_worst:",corr_radius_worst)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_tex_worst = df['texture_worst'].corr(diag)
print("correlation between diagnosis and texture_worst:",corr_tex_worst)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_perimeter_worst = df['perimeter_worst'].corr(diag)
print("correlation between diagnosis and Concave Points_se:",corr_perimeter_worst)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_area_worst = df['area_worst'].corr(diag)
print("correlation between diagnosis and area_worst:",corr_area_worst)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_smoothness_worst = df['smoothness_worst'].corr(diag)
print("correlation between diagnosis and smoothness_worst:",corr_smoothness_worst)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_compactness_worst = df['compactness_worst'].corr(diag)
print("correlation between diagnosis and compactness_worst:",corr_compactness_worst)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_concavity_worst = df['concavity_worst'].corr(diag)
print("correlation between diagnosis and Concavity worst:",corr_concavity_worst)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_cp_worst= df['concave points_worst'].corr(diag)
print("correlation between diagnosis and Concave Points_worst:",corr_cp_worst)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_symmetry_worst = df['symmetry_worst'].corr(diag)
print("correlation between diagnosis and symmetry_worst:",corr_symmetry_worst)

#correlation between diagnosis and texture_mean
diag = df1['diagnosis']
corr_fd_worst = df['fractal_dimension_worst'].corr(diag)
print("correlation between diagnosis and Fractal Dimension:",corr_fd_worst)


print("\n\n")

##############################################################################################################################
'''
#X and y values
X = df1.iloc[:,3:32]
y = df1['diagnosis']

#Decision tree classifier
print("Decision Tree Classifier")
model = DecisionTreeClassifier()
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
model = model.fit(X,y)                                      #change to X,y
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("Decision Tree Classifier = ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))    #change to X,y
print("\n\n")


#random forest
print("Random forest Classifier")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02 
    )
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
model = model.fit(X,y)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("Random Forest = ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")


#knn
print("KNN")
model = KNeighborsClassifier(n_neighbors=3)
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
model = model.fit(X,y)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("KNN= ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")


#Logistic Regression
print("Logistic Regression")
model = LogisticRegression()
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
model = model.fit(X,y)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("Logistic Regression = ", rf_roc_auc)
#print(metrics.classification_report(y_test,model.predict(X_test)))
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")

#ADA classifier
print("ADA classifier")
model = ada = AdaBoostClassifier(n_estimators=400)
model = model.fit(X,y)
kfold = ms.StratifiedKFold(n_splits=10)
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("ADA Classifier = ", rf_roc_auc)
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
#print(metrics.classification_report(y_test,model.predict(X_test)))
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")


#naive bayes classifier
print("Naive bayes")
model = NB.GaussianNB()
model = model.fit(X,y)
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
kfold = ms.StratifiedKFold(n_splits=10)
#print(metrics.classification_report(y_test,model.predict(X_test)))
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("Naive Bayes Classifier = ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")


print("SVM")
model = SVC()
model = model.fit(X,y)
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
kfold = ms.StratifiedKFold(n_splits=10)
#print(metrics.classification_report(y_test,model.predict(X_test)))
rf_roc_auc = roc_auc_score(y, model.predict(X))
print ("SVM = ", rf_roc_auc)
predCV = ms.cross_val_predict(model, X, y, cv=kfold)
print("prediction cross validation", predCV)                                
precisionVal = metrics.precision_score(y,predCV)
print("Precision value", precisionVal)
recallVal = metrics.recall_score(y,predCV)
print("Recall value", recallVal)
f1Val = metrics.f1_score(y,predCV)
print("f1 value", f1Val)
KappaVal = metrics.cohen_kappa_score(y, predCV)
print("Kappa value", KappaVal)
Accuracy= metrics.accuracy_score(y,predCV )
print("Accuracy", Accuracy)
print(metrics.classification_report(y,model.predict(X)))
print("\n\n")
##################################################################################################################
#important feature in the dataset
X = df1.iloc[:,1:32]
y = df1['diagnosis']

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,max_features=None,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking for all factors:")

for f in range(X.shape[1]):
    print("%d. feature %d : %f" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest 
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
