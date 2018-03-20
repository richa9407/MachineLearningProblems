
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().magic('matplotlib inline')


data=pd.read_csv('Data_sets/carTrainData.csv')


data.head()


data.info()

for i in data.columns:
    print(data[i].unique(),"\t",data[i].nunique())

for i in data.columns:
    print(data[i].value_counts())
    print()

sns.countplot(data['V7'])

for i in data.columns[:-1]:
    plt.figure(figsize=(12,6))
    plt.title("For feature '%s'"%i)
    sns.countplot(data[i],hue=data['V7'])

from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()


for i in data.columns:
    data[i]=le.fit_transform(data[i])


data.head()

fig=plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True)

X=data[data.columns[:-1]]
y=data['V7']

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logreg=LogisticRegression(solver='newton-cg',multi_class='multinomial')

logreg.fit(X_train,y_train)

pred=logreg.predict(X_test)

logreg.score(X_test,y_test)

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_jobs=-1,random_state=51)

from sklearn.metrics import f1_score

rfc.fit(X_train,y_train)
print(rfc.score(X_test,y_test))
print(f1_score(y_test, rfc.predict(X_test), average='macro'))

print(classification_report(y_test,rfc.predict(X_test)))

from sklearn.grid_search import GridSearchCV

param_grid={'criterion':['gini','entropy'],
           'max_depth':[2,5,10,20],
           'max_features':[2,4,6,'auto'],
           'max_leaf_nodes':[2,3,None],}

grid=GridSearchCV(estimator=RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=51),
                  param_grid=param_grid,cv=10,n_jobs=-1)

grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.best_score_)

print(X.columns)
print(rfc.feature_importances_)

from sklearn.model_selection import learning_curve

lc=learning_curve(RandomForestClassifier(n_estimators=50,criterion='entropy',max_features=6,max_depth=10,random_state=51,
                                             max_leaf_nodes=None,n_jobs=-1,),X_train,y_train,cv=5,n_jobs=-1)
size=lc[0]
train_score=[lc[1][i].mean() for i in range (0,5)]
test_score=[lc[2][i].mean() for i in range (0,5)]
fig=plt.figure(figsize=(12,8))
plt.plot(size,train_score)
plt.plot(size,test_score)

from sklearn.model_selection import validation_curve

param_range=[10,25,50,100]
curve=validation_curve(rfc,X_train,y_train,cv=5,param_name='n_estimators',
    param_range=param_range,n_jobs=-1)

train_score=[curve[0][i].mean() for i in range (0,len(param_range))]
test_score=[curve[1][i].mean() for i in range (0,len(param_range))]
fig=plt.figure(figsize=(6,8))
plt.plot(param_range,train_score)
plt.plot(param_range,test_score)
plt.xticks=param_range

param_range=range(1,len(X.columns)+1)
curve=validation_curve(RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=51),X_train,y_train,cv=5,
    param_name='max_features',param_range=param_range,n_jobs=-1)

train_score=[curve[0][i].mean() for i in range (0,len(param_range))]
test_score=[curve[1][i].mean() for i in range (0,len(param_range))]
fig=plt.figure(figsize=(6,8))
plt.plot(param_range,train_score)
plt.plot(param_range,test_score)
plt.xticks=param_range

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

#Predicting on traing data
cnf_matrix=confusion_matrix(y_test,rfc.predict(X_test))
np.set_printoptions(precision=2)

print('::RESULTS FOR TRAINING DATASET::')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=data,
                      title='Confusion matrix, without normalization')


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=data ,normalize= True , title= 'Normalized Confusion Matrix')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y_test,
                      title='Confusion matrix, without normalization')


plt.figure()
plot_confusion_matrix(cnf_matrix, classes= y_test,normalize= True , title= 'Normalized Confusion Matrix')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y_train,
                      title='Confusion matrix, without normalization')


plt.figure()
plot_confusion_matrix(cnf_matrix, classes= y_train,normalize= True , title= 'Normalized Confusion Matrix')

plt.show()


#Predictions on test data
test_data=pd.read_csv('Data_sets/carTestData.csv')

for i in data.columns:
    data[i]=le.fit_transform(data[i])

Xt = data[data.columns[:-1]]
Yt = data['V7']

cnf_matrix=confusion_matrix(Yt,rfc.predict(Xt))
np.set_printoptions(precision=2)

print('\n::RESULTS FOR TEST DATASET::')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=data,
                      title='Confusion matrix, without normalization')


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=data ,normalize= True , title= 'Normalized Confusion Matrix')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y_test,
                      title='Confusion matrix, without normalization')


plt.figure()
plot_confusion_matrix(cnf_matrix, classes= y_test,normalize= True , title= 'Normalized Confusion Matrix')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y_train,
                      title='Confusion matrix, without normalization')


plt.figure()
plot_confusion_matrix(cnf_matrix, classes= y_train,normalize= True , title= 'Normalized Confusion Matrix')

plt.show()