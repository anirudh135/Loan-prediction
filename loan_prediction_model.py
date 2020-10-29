import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv('../loan_prediction/test_Y3wMUE5_7gLdaTN.csv')
df = pd.read_csv("../loan_prediction/train_u6lujuX_CVtuZ9i.csv")

data['LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].mean())
data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].median())
df.dropna(inplace=True)
df['Loan_Status'].replace('N',0,inplace=True)
df['Loan_Status'].replace('Y',1,inplace=True)
df2=df.drop(labels=['ApplicantIncome'],axis=1)
df2=df2.drop(labels=['CoapplicantIncome'],axis=1)
df2=df2.drop(labels=['LoanAmount'],axis=1)
df2=df2.drop(labels=['Loan_Amount_Term'],axis=1)
df2=df2.drop(labels=['Loan_ID'],axis=1)
le=LabelEncoder()
ohe=OneHotEncoder()
df2['Property_Area']=le.fit_transform(df2['Property_Area'])
df2['Dependents']=le.fit_transform(df2['Dependents'])
df2=pd.get_dummies(df2)
df2=df2.drop(labels=['Gender_Female'],axis=1)
df2=df2.drop(labels=['Married_No'],axis=1)
df2=df2.drop(labels=['Education_Not Graduate'],axis=1)
df2=df2.drop(labels=['Self_Employed_No'],axis=1)
df2=df2.drop('Self_Employed_Yes',1)
df2=df2.drop('Dependents',1)
df2=df2.drop('Education_Graduate',1)
X=df2.drop('Loan_Status',1)
Y=df2['Loan_Status']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=6)
log=LogisticRegression()
log.fit(x_train,y_train)
x1 = log.predict(x_test)
print(accuracy_score(y_test,x1))
print(x_train)

pickle.dump(log,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

