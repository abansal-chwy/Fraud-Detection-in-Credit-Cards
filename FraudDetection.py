
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

data=pd.read_csv("creditcard.csv")

#Check the data
print(data.shape)  #to get the number of rows and columns
#284807 rows and 31 columns

print(data.head()) #get first 5 rows of the data
#Reduced by PCA

print(data.describe())
#min time = 0 max time = 172792 (2 days)

#check missing data
print(data.isnull().sum())
#no missing data

#Visualize class 0/1 or fraudulent or not
classes=[0,1]
classes_count=[data['Class'].value_counts().values[0],data['Class'].value_counts().values[1]]
plt.bar(classes,classes_count,align='center')
plt.title('Credit Card Fraud Class')
plt.xticks("0","1")
plt.xlabel("class")
plt.ylabel("counts")
plt.show()

#Visualize transactions over time..
sns.kdeplot(data.loc[data['Class']==0]['Time'],label="class 0")
sns.kdeplot(data.loc[data['Class']==1]['Time'],label="class 1")
plt.title('Credit Card Transactions per Time')
plt.legend()
plt.show()


#Check for outliers in amount
data_to_plot=[data.loc[data['Class']==0]['Amount'],data.loc[data['Class']==1]['Amount']]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_to_plot)

for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)
                
## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)
## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
ax.set_xticklabels(['Class 0', 'Class 1'])
## Remove top axes and right axes ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()              
plt.legend()
plt.show()


#Scatter plot for fraud(Amount) vs Time
plt.scatter(data.loc[data['Class']==1]['Time'],data.loc[data['Class']==1]['Amount'])
plt.xlabel("Time")
plt.ylabel("Amount")
plt.show()


#Features co-relation
plt.figure(figsize = (14,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()



#Split the data into train and test
X=data.iloc[:,0:30].values
y=data.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0,shuffle=True) 


#Applying Random forest
clf = RandomForestClassifier(max_depth=50, random_state=0)
clf.fit(X_train, y_train)
y_pred_RF=clf.predict(X_test)
cm=confusion_matrix(y_test,y_pred_RF)
print(roc_auc_score(y_test, y_pred_RF))
#Area Under the Curve=86.7%
#Accuracy=99.94

#Applying SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_SVM = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred_SVM)
print(roc_auc_score(y_test, y_pred_RF))
#86.7%
#Accuracy=99.9%