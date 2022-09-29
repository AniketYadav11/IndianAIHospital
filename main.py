"""Breast Cancer Detection Using Machine Learning """
# import libraries
import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization


#Load breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()


# name of features

# print(cancer_dataset['feature_names'])

# location/path of data file
# print(cancer_dataset['filename'])

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print(cancer_dataset)
# create datafrmae
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
             columns = np.append(cancer_dataset['feature_names'], ['target']))

# print(cancer_df.head(5))
# Information  of cancer Dataframe
# print(cancer_df.info())

# Description of dataset
# print(cancer_dataset['DESCR'])

# Paiplot of cancer dataframe
# sns.pairplot(cancer_df,hue='target')
# plt.show()

# create second DataFrame by droping target
cancer_df2 = cancer_df.drop(['target'], axis = 1)
# print("The shape of 'cancer_df2' is : ", cancer_df2.shape)

#input variables
X = cancer_df.drop(['target'],axis=1)
# print(X.head(6))

# Output variables
y = cancer_df['target']
# print(y.head(4))
print(y.shape)


# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)

"""Feature Scaling """
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


"""Machine Learning Model Building """
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

"""SUPORT VECTOR CLASSIFIER"""
# from sklearn.svm import SVC
# svc_classifier = SVC()
# svc_classifier.fit(X_train_sc,y_train)
# y_pred_svc = svc_classifier.predict(X_test_sc)
# print(accuracy_score(y_test,y_pred_svc))

# from sklearn.svm import SVC
# svc_classifier = SVC()
# svc_classifier.fit(X_train,y_train)
# y_pred_svc = svc_classifier.predict(X_test)
# print(accuracy_score(y_test,y_pred_svc))
""" For features scale data Accuracy 96.4% """
""" Accuracy = 93 %"""

""" Logistic Regression """

# from sklearn.linear_model import LogisticRegression
# lr_classifier = LogisticRegression(random_state = 51, penalty = 'l2')
# lr_classifier.fit(X_train_sc, y_train)
# y_pred_lr = lr_classifier.predict(X_test_sc)
# print(accuracy_score(y_test, y_pred_lr))

# from sklearn.linear_model import LogisticRegression
# lr_classifier = LogisticRegression(random_state = 51, penalty = 'l2')
# lr_classifier.fit(X_train, y_train)
# y_pred_lr = lr_classifier.predict(X_test)
# print(accuracy_score(y_test, y_pred_lr))
"""For feature scale data Accuracy  97.3 %"""
"""Accuracy = 95%"""

"""K – Nearest Neighbor Classifier"""
# from sklearn.neighbors import KNeighborsClassifier
# knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# knn_classifier.fit(X_train, y_train)
# y_pred_knn = knn_classifier.predict(X_test)
# print(accuracy_score(y_test, y_pred_knn))

# from sklearn.neighbors import KNeighborsClassifier
# knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# knn_classifier.fit(X_train_sc, y_train)
# y_pred_knn = knn_classifier.predict(X_test_sc)
# print(accuracy_score(y_test, y_pred_knn))
"""For featured scale data Accuracy 96.4 %"""
"""Accuracy = 93%"""

"""Naive Bayes Classifier """
# from sklearn.naive_bayes import GaussianNB
# nb_classifier = GaussianNB()
# nb_classifier.fit(X_train, y_train)
# y_pred_nb = nb_classifier.predict(X_test)
# print(accuracy_score(y_test, y_pred_nb))

# from sklearn.naive_bayes import GaussianNB
# nb_classifier = GaussianNB()
# nb_classifier.fit(X_train_sc, y_train)
# y_pred_nb = nb_classifier.predict(X_test_sc)
# print(accuracy_score(y_test, y_pred_nb))
"""for featured data Accuracy  93.8 %"""
"""Accuracy = 94%"""

"""Decision Tree Classifier"""
# from sklearn.tree import DecisionTreeClassifier
# dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
# dt_classifier.fit(X_train, y_train)
# y_pred_dt = dt_classifier.predict(X_test)
# print(accuracy_score(y_test, y_pred_dt))

# from sklearn.tree import DecisionTreeClassifier
# dt_classifier = DecisionTreeClassifier(criterion='entropy',random_state=51)
# dt_classifier.fit(X_train_sc,y_train)
# y_pred_dt = dt_classifier.predict(X_test_sc)
# print(accuracy_score(y_test,y_pred_dt))
"""For featured data Accuracy 94.7% """
"""Accuracy = 94.2%"""

"""Random Forest Classifier"""
# from  sklearn.ensemble import RandomForestClassifier
# rf_classifier = RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=51)
# rf_classifier.fit(X_train,y_train)
# y_pred_rf = rf_classifier.predict(X_test)
# print(accuracy_score(y_test,y_pred_rf))

# from sklearn.ensemble import  RandomForestClassifier
# rf_classifier = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=51)
# rf_classifier.fit(X_train_sc,y_train)
# y_pred_rf = rf_classifier.predict(X_test_sc)
# print(accuracy_score(y_test,y_pred_rf))
"""for Featured data Accuracy 97.3%"""
"""Accuracy = 97%"""

"""Adaboost Classifier"""
from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(n_estimators=2000,learning_rate=0.1,algorithm='SAMME.R',random_state=1)
adb_classifier.fit(X_train,y_train)
y_pred_adb = adb_classifier.predict(X_test)
print(accuracy_score(y_test,y_pred_adb))

# from sklearn.ensemble import AdaBoostClassifier
# adb_classifier = AdaBoostClassifier( n_estimators=200,learning_rate=0.1,algorithm='SAMME.R',random_state=1)
# adb_classifier.fit(X_train_sc,y_train)
# y_pred_adb = adb_classifier.predict(X_test_sc)
# print(accuracy_score(y_test,y_pred_adb))
"""For featured data Accuracy 98.2% """
"""Accuracy = 98%"""

# HyperParameter tuning
# grid = dict()
# grid['n_estimators'] = [100,500,1000,1500]
# grid['learning_rate'] = [0.0001,0.001,0.01, 0.1, 1.0]
# grid['algorithm']  = ['SAMME','SAMME.R']
#
# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(estimator=AdaBoostClassifier(),param_grid=grid,n_jobs=-1,cv=10, scoring='accuracy')
# grid_result = grid_search.fit(X_train,y_train)
# print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))

adb_classifier_pt = AdaBoostClassifier(n_estimators=500, learning_rate=0.1, algorithm='SAMME')
adb_classifier_pt.fit(X_train,y_train)
y_pred_adb_pt = adb_classifier_pt.predict(X_test)



# Confusion Matrix
# cm = confusion_matrix(y_test,y_pred_adb_pt)
# plt.title("Heatmap of Confusion Matrix ", fontsize=15)
# sns.heatmap(cm, annot=True)
# plt.show()

#Classification Report of Model
# print(classification_report(y_test,y_pred_adb_pt))
# # Cross validatation
# from sklearn.model_selection import  cross_val_score
# cross_validation = cross_val_score(estimator=adb_classifier_pt,X= X_train_sc, y = y_train, cv=10)
# print("Cross validation accuracy of AdaBoost model = ",cross_validation)
# print("\nCross validation mean accuracy of AdaBoost model = ",cross_validation.mean())

# Pickle
import pickle

# save model
pickle.dump(adb_classifier_pt, open('breast_cancer_detector.pickle', 'wb'))

# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)

# confusion matrix
print('Confusion matrix of Adaboost Classifier model: \n', confusion_matrix(y_test, y_pred), '\n')

# show the accuracy
print('Accuracy of Adaboost Classifier model = ', accuracy_score(y_test, y_pred))
