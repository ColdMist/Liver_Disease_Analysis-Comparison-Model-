# %% 1
# Package imports
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from utilities import *

dataset = pd.read_csv('Dataset/Indian Liver Patient Dataset (ILPD).csv')
#dataset = pd.read_csv('Dataset/Imputed_data.csv')
#dataset = pd.read_csv('Dataset/my_data.csv')
dataset = pd.DataFrame(dataset)
#dataset ['Problem'] = dataset['Problem'].astype('category')
#Encode the categorical variables
label_encoder = LabelEncoder()
dataset['Problem'] = label_encoder.fit_transform(dataset['Problem'])
dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])
#Find the highly correlated variables
corr_value=get_higly_correlated_matrix(dataset)

dataset['Age'] = remove_outlier(dataset, 'Age')
dataset['TB'] = remove_outlier(dataset, 'TB')
dataset['DB'] = remove_outlier(dataset, 'DB')
dataset['Alkphos'] = remove_outlier(dataset, 'Alkphos')
dataset['Sgpt '] = remove_outlier(dataset, 'Sgpt ')
dataset['Sgot'] = remove_outlier(dataset, 'Sgot')
dataset['TP'] = remove_outlier(dataset, 'TP')
dataset['ALB'] = remove_outlier(dataset, 'ALB')
dataset['A/G'] = remove_outlier(dataset, 'A/G')

#dataset = miceImputation(pd.DataFrame(dataset))
#Find the highly correlated features

#Remove highly correlated variables over a threshold
X = dataset.iloc[:,[0,1,3,4,5,8,9]].values
X = miceImputation(X)
#imp = Imputer(missing_values="NaN", strategy = 'mean', axis = 0)
y = dataset.iloc[:,10].values
#Scale specific features
X = scaleMinMax(X)
'''
showBoxPlot(X[:,[6]])
showBoxPlot(X[:,[2]])
show_histogram(X[:,[2]])
'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
# Fitting classification methods to the Training sets
from sklearn.tree import DecisionTreeClassifier
classifierDtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
classifierDtree.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifierRandomForest = RandomForestClassifier(n_estimators=1000,random_state=0)
classifierRandomForest.fit(X_train,y_train)
#Show Random Forest Importance
#RandomForestImportanceShow()
#print(X_train.Shape)
from sklearn import svm
classifierSVM = svm.SVC(kernel='rbf',gamma=0.1, probability=True )
classifierSVM.fit(X_train,y_train)

classifierMLP = MLPClassifier()
classifierMLP.fit(X_train,y_train)

classifierGaussianNB = GaussianNB()
classifierGaussianNB.fit(X_train, y_train)

classifierKNN = KNeighborsClassifier(n_neighbors=2)
classifierKNN.fit(X_train,y_train)

classifierLogisticReg = sklearn.linear_model.LogisticRegressionCV()
classifierLogisticReg.fit(X_train, y_train)


from keras.optimizers import RMSprop
#opt = RMSprop(lr=0.0001, decay=1e-6)
opt = 'adam'
def create_baseline():
    model = Sequential()
    model.add(Dense(units = 100,  activation = 'relu',  input_dim =7))
    model.add(Dense(units = 100, activation = 'relu'))
    model.add(Dense(units = 150, activation = 'relu', ))
    #model.add(Dense(units = 1, activation = 'relu'))
    model.add(Dense(units =50,  activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
#neural Network training
classifierANN = create_baseline()
classifierANN.fit(X_train, y_train, epochs=1000,batch_size=64)
y_pred_ANN = classifierANN.predict(X_test)
y_pred_ANN = (y_pred_ANN > 0.5)
scores = classifierANN.evaluate(X_train, y_train)
ANN_filter = np.where(y_pred_ANN > 0.5,1,0)

score, acc = classifierANN.evaluate(X_test, y_test,
                            batch_size=32)
##################
# Predicting the Test set results
y_pred_Dtree = classifierDtree.predict(X_test)
y_pred_RandomForest = classifierRandomForest.predict(X_test)
y_pred_SVM = classifierSVM.predict(X_test)
y_pred_MLP = classifierMLP.predict(X_test)
y_pred_GaussianNB = classifierGaussianNB.predict(X_test)
y_pred_KNN = classifierKNN.predict(X_test)
y_pred_logistic = classifierLogisticReg.predict(X_test)
# Get the accuracy
#print ('accuracy: TRAINING', classifier.score(X_train,y_train))
#print ('accuracy: TESTING', classifier.score(X_test,y_test))
print ('accuracy: TRAINING (DTREE)', classifierDtree.score(X_train,y_train))
print('accuracy: TESTING (DTREE) ', classifierDtree.score(X_test,y_test))
print ('accuracy: TRAINING (RFOREST)', classifierRandomForest.score(X_train,y_train))
print('accuracy: TESTING (RFOREST) ',classifierRandomForest.score(X_test,y_test))
print ('accuracy: TRAINING (SVM)', classifierSVM.score(X_train,y_train))
print('accuracy: TESTING (SVM) ', classifierSVM.score(X_test,y_test))
print ('accuracy: TRAINING (MLP)', classifierMLP.score(X_train,y_train))
print('accuracy: TESTING (MLP) ', classifierMLP.score(X_test,y_test))
print ('accuracy: TRAINING (GNB)', classifierGaussianNB.score(X_train,y_train))
print('accuracy: TESTING (GNB) ', classifierGaussianNB.score(X_test,y_test))
print ('accuracy: TRAINING (KNN)', classifierKNN.score(X_train,y_train))
print('accuracy: TESTING (KNN) ', classifierKNN.score(X_test,y_test))
print("accuracy: TRAINING (ANN): %.2f%%" % (max(scores)*100))
print('accuracy: TESTING (ANN) ', acc)
#print("accuracy: TESTING (ANN)%s: %.2f%%" % (classifierANN.metrics_names[1], score[1]*100))
print("accuracy: TRAINING (Logistic Regression) " , (classifierLogisticReg.score(X_train,y_train)))
print('accuracy: TESTING (Logistic Regression) ', classifierLogisticReg.score(X_test, y_test))

from sklearn.metrics import roc_curve, auc
y_predict_probabilities_Dtree = classifierDtree.predict_proba(X_test)[:,1]
y_predict_probabilities_RandomForest = classifierRandomForest.predict_proba(X_test)[:,1]
y_predict_probabilities_SVM = classifierSVM.predict_proba(X_test)[:,1]
y_predict_probabilities_MLP = classifierMLP.predict_proba(X_test)[:,1]
y_predict_probabilities_GaussianNB = classifierGaussianNB.predict_proba(X_test)[:,1]
y_predict_probabilities_KNN = classifierKNN.predict_proba(X_test)[:,1]
y_predict_probabilities_ANN = classifierANN.predict_proba(X_test)
y_predict_probabilities_LogReg = classifierLogisticReg.predict_proba(X_test)[:,1]

fpr_Dtree, tpr_Dtree, _ = roc_curve(y_test, y_predict_probabilities_Dtree)
roc_auc_Dtree = auc(fpr_Dtree, tpr_Dtree)
print("Area under the curve Decision Tree: ", roc_auc_Dtree)

fpr_RandomForest, tpr_RandomForest, _ = roc_curve(y_test, y_predict_probabilities_RandomForest)
roc_auc_RandomForest = auc(fpr_RandomForest, tpr_RandomForest)
print("Area under the curve Random Forest: ", roc_auc_RandomForest)

fpr_SVM, tpr_SVM, _ = roc_curve(y_test, y_predict_probabilities_SVM)
roc_auc_SVM = auc(fpr_SVM, tpr_SVM)
print("Area under the curve Support Vector Machine: ", roc_auc_SVM)

fpr_MLP, tpr_MLP, _ = roc_curve(y_test, y_predict_probabilities_MLP)
roc_auc_MLP = auc(fpr_MLP, tpr_MLP)
print("Area under the curve Multilayer Perceptron: " , roc_auc_MLP)

fpr_GaussianNB, tpr_GaussianNB, _ = roc_curve(y_test, y_predict_probabilities_GaussianNB)
roc_auc_GaussianNB = auc(fpr_GaussianNB, tpr_GaussianNB)
print("Area under the curve Gaussian Naive Bayes: ", roc_auc_GaussianNB)

fpr_KNN, tpr_KNN, _ = roc_curve(y_test, y_predict_probabilities_KNN)
roc_auc_KNN = auc(fpr_KNN, tpr_KNN)
print("Area under the curve K Nearest Neigbor: ", roc_auc_KNN)

fpr_ANN, tpr_ANN, _ = roc_curve(y_test, y_predict_probabilities_ANN)
roc_auc_ANN = auc(fpr_ANN, tpr_ANN)
print("Area under the curve Artificial Neural Network: ", roc_auc_ANN)

fpr_LogReg, tpr_LogReg, _ = roc_curve(y_test, y_predict_probabilities_LogReg)
roc_auc_LogReg = auc(fpr_LogReg, tpr_LogReg)
print("Area under the curve Logistic Regression: ", roc_auc_LogReg)

plt.figure()
plt.plot(fpr_Dtree, tpr_Dtree, color='darkorange',
         lw=2, label='Decision Tree (area = %0.2f)' % roc_auc_Dtree)
plt.plot(fpr_RandomForest, tpr_RandomForest, color='darkgreen',
         lw=2, label='Random Forest Classification (area = %0.2f)' % roc_auc_RandomForest)
plt.plot(fpr_SVM, tpr_SVM, color='red',
         lw=2, label='SVM (area = %0.2f)' % roc_auc_SVM)
plt.plot(fpr_MLP, tpr_MLP, color='blue',
         lw=2, label='MLP (area = %0.2f)' % roc_auc_MLP)
plt.plot(fpr_GaussianNB, tpr_GaussianNB, color='purple',
         lw=2, label='GaussianNB (area = %0.2f)' % roc_auc_GaussianNB)
plt.plot(fpr_KNN, tpr_KNN, color='grey',
         lw=2, label='KNN (area = %0.2f)' % roc_auc_KNN)
plt.plot(fpr_ANN, tpr_ANN, color='black',
         lw=2, label='ANN (area = %0.2f)' % roc_auc_ANN)
plt.plot(fpr_LogReg, tpr_LogReg, color='orange',
         lw=2, label='LogReg (area = %0.2f)' % roc_auc_LogReg)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right", fontsize=7)
plt.show()#####code will be executed acco
