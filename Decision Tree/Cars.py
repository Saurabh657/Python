#Implementation 


# In[34]:


import numpy as np
import pandas as pd


# In[35]:


cars_data = pd.read_csv("cars.csv",header = None)
cars_data.head()


# In[36]:


cars_data.columns=['buying','maint','doors','persons','lug_boot','safety','classes']


# In[37]:


cars_data.head()


# In[38]:


cars_data.isnull().sum()


# In[39]:


cars_df=pd.DataFrame.copy(cars_data)


# In[40]:


colname=cars_df.columns
colname


# In[41]:


from sklearn import preprocessing

le=preprocessing.LabelEncoder()

for x in colname:
    cars_df[x]=le.fit_transform(cars_df[x])
    


# In[42]:


cars_df.head()


# In[43]:


cars_data.classes.value_counts()

#acc=0
#good=1
#unacc=2
#vgood=3


# In[44]:


X=cars_df.values[:,:-1]
Y=cars_df.values[:,-1]
Y=Y.astype(int)


# In[45]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)

X=scaler.transform(X)


# In[46]:


from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
random_state=10)


# # Running the Decision Tree Model

# In[47]:


from sklearn.tree import DecisionTreeClassifier

model_DecisionTree = DecisionTreeClassifier(criterion='gini',
                                            random_state=10)

model_DecisionTree.fit(X_train,Y_train)


# In[48]:


Y_pred= model_DecisionTree.predict(X_test)
print(Y_pred)


# In[49]:


print(list(zip(cars_df.columns,model_DecisionTree.feature_importances_)))


# In[50]:


#cars_df.drop(["doors","persons"],axis = 1,inplace= True)


# # IMPLEMENTING DECISION TREE

# In[51]:



from sklearn import tree
with open("model_DecisionTree1.txt", "w") as f:
    f = tree.export_graphviz(model_DecisionTree, feature_names=cars_df.columns[:-1],
                                            out_file=f)
   

#generate the file and upload the code in webgraphviz.com to plot the decision tree


# In[52]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("CLASSIFICATION MATRIX:")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("ACCURACY OF THE MODEL:",acc)


# # IMPLEMENTING KNN

# In[53]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors = 8,
metric='euclidean')
#fit the model on the data and predict the values
model_KNN.fit(X_train,Y_train)

Y_pred=model_KNN.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[54]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("CLASSIFICATION MATRIX:")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("ACCURACY OF THE MODEL:",acc)


# # IMPLEMENTING LOGISTIC

# In[55]:


from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))



# In[56]:


# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("CLASSIFICATION MATRIX:")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("ACCURACY OF THE MODEL:",acc)


# # IMPLEMENTATION USING BAGGING ENSEMBLE

# In[57]:


#predicting using the Bagging_Classifier
from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier(n_estimators=100,random_state=10)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)


# In[58]:


cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("CLASSIFICATION MATRIX:")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("ACCURACY OF THE MODEL:",acc)


# In[ ]:





# In[59]:


#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(n_estimators=5000, max_depth=8,
                                           min_samples_leaf=5,random_state=10)

#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred=model_RandomForest.predict(X_test)


# In[60]:


cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("CLASSIFICATION MATRIX:")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("ACCURACY OF THE MODEL:",acc)


# In[ ]:





# In[61]:


#predicting using the AdaBoost_Classifier
from sklearn.ensemble import AdaBoostClassifier

model_AdaBoost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=10),
n_estimators=100,
random_state=10)
#fit the model on the data and predict the values
model_AdaBoost.fit(X_train,Y_train)
Y_pred=model_AdaBoost.predict(X_test)


# In[62]:


cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("CLASSIFICATION MATRIX:")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("ACCURACY OF THE MODEL:",acc)


# In[ ]:





# In[63]:


#predicting using the Gradient_Boosting_Classifier
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier(n_estimators=150,
random_state=10)

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)


# In[64]:


cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("CLASSIFICATION MATRIX:")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("ACCURACY OF THE MODEL:",acc)


# In[ ]:





# In[65]:


from xgboost import XGBClassifier

model_GradientBoosting=XGBClassifier(random_state=10)

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)


# In[66]:


cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print("CLASSIFICATION MATRIX:")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("ACCURACY OF THE MODEL:",acc)


# In[67]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# create the sub models
estimators = []
#model1 = LogisticRegression()
#estimators.append(('log', model1))
model2 = DecisionTreeClassifier(criterion='gini',random_state=10)
estimators.append(('cart', model2))
model3 = SVC(kernel="rbf", C=50,gamma=0.1)
estimators.append(('svm', model3))
#model4 = KNeighborsClassifier(n_neighbors=8, metric='euclidean')
#estimators.append(('knn', model4))


# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#print(Y_pred)


# In[ ]:




