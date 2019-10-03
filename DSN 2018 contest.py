# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:52:29 2018

@author: Michael & Cyril
"""

#importing the relevant libraries\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

#importing the dataset
dataset=pd.read_csv(r"C:\Users\CYRIL\Downloads\DSN 2018\DSN_train.csv")
testset=pd.read_csv(r"C:\Users\CYRIL\Downloads\DSN 2018\DSN_test.csv")

#evaluate missing values6
dataset.isnull().sum()
missing_percentage= (dataset.isnull().sum()/len(dataset))*100
missing_percentage=missing_percentage.drop(missing_percentage[missing_percentage==0].index).sort_values(ascending=False)

plt.clf()
plt.figure()
plt.bar(missing_percentage.index,missing_percentage)
#plt.xticks(rotation='90')
plt.xlabel('Index of values that are missing')
plt.ylabel('Percentage of missing values')
plt.show() 

#data preprocessing
X=dataset.iloc[:,[1,3,4,5,6,7,10,11]].values
y=dataset.iloc[:,-1].values
label_encoder1=LabelEncoder()
X[:,0]=label_encoder1.fit_transform(X[:,0])
X[:,2]=LabelEncoder().fit_transform(X[:,2])
X[:,4]=LabelEncoder().fit_transform(X[:,4])
X[:,6]=LabelEncoder().fit_transform(X[:,6])
X[:,7]=LabelEncoder().fit_transform(X[:,7])
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
hot_encoder=OneHotEncoder(categorical_features=[0,2,4,6,7])
X=hot_encoder.fit_transform(X).toarray()
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=7,test_size=0.2)
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
#Sscaling the big guy
X=StandardScaler().fit_transform(X)


xgb_reg=xgb.XGBRegressor(objecive='reg:linear',colsample_bytree=0.3,learning_rate=0.1,
                         n_estimators=10,max_depth=5,alpha=10)
xgb_reg.fit(x_train,y_train,verbose=True)

y_pred=xgb_reg.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))


#getting ready the test file
#data preprocessing
testset=pd.read_csv(r"C:\Users\CYRIL\Downloads\DSN 2018\DSN_test.csv")
A=testset.iloc[:,[1,3,4,5,6,7,10,11]].values
label_encoder1=LabelEncoder()
A[:,0]=label_encoder1.fit_transform(A[:,0])
A[:,2]=LabelEncoder().fit_transform(A[:,2])
A[:,4]=LabelEncoder().fit_transform(A[:,4])
A[:,6]=LabelEncoder().fit_transform(A[:,6])
A[:,7]=LabelEncoder().fit_transform(A[:,7])
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(A[:,1:3])
A[:,1:3]=imputer.transform(A[:,1:3])
hot_encoder=OneHotEncoder(categorical_features=[0,2,4,6,7])
A=hot_encoder.fit_transform(A).toarray()
A=StandardScaler().fit_transform(A)
X=StandardScaler().fit_transform(X)


#retrain the model on all the x_data
xgb_full_reg=xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.3,learning_rate=0.1,
                              n_estimators=10,max_depth=5,alpha=10)
xgb_full_reg.fit(X,y,verbose=True)
B_preds=xgb_full_reg.predict(A)

B_preds=pd.DataFrame(B_preds,columns=['Product_Supermarket_Sales'])
submission_file=pd.DataFrame(testset['Product_Supermarket_Identifier'],columns=['Product_Supermarket_Identifier'])
submission_file['Product_Supermarket_Sales']=B_preds
submission_file.to_csv('results.csv', header=True, index=False)

#play=pd.read_csv('results.csv')
#play=play.iloc[:,1:3]
#play.to_csv('real_results.csv')