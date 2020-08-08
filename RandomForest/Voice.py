
#Import the necessary libraries
import numpy as np
import pandas as pd


#read the data into a dataframe
df = pd.read_csv(r"C:\Users\Saurabh\Projects\voice.csv",header = 0)
df.head()


#Checking the missing values
df.isnull().sum()



#checking the datatypes
df.dtypes



colname=df.columns
colname



#converting the categorical variables into numericals by Label Encoder
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

for x in colname:
    df[x]=le.fit_transform(df[x])



df.head()



df.label.value_counts()



#Naming the independent and dependant variables
X=df.values[:,:-1]
Y=df.values[:,-1]
Y=Y.astype(int)




#Scaling the X variable
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)

X=scaler.transform(X)



#Spliting the data into Test and Train
from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
random_state=10)



#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(n_estimators=5000, max_depth=8,
                                           min_samples_leaf=5,random_state=10)

#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)




from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("CLASSIFICATION MATRIX:")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("ACCURACY OF THE MODEL:",acc)






