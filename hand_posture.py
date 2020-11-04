import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
import warnings
warnings.filterwarnings("ignore")

#libray for all the classifier and the gridsearc for cross validation is obtained from sklearn website


def feature_extraction(data,train):
    dataX=data[['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']]
    meanX=dataX.mean(axis = 1, skipna = True) 
    dataY=data[['Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Y10','Y11']]
    meanY=dataY.mean(axis = 1, skipna = True) 
    dataZ=data[['Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','Z10','Z11']]
    meanZ=dataZ.mean(axis = 1, skipna = True) 
    stdX=dataX.std(axis = 1, skipna = True)
    stdY=dataY.std(axis = 1, skipna = True)
    stdZ=dataZ.std(axis = 1, skipna = True)
    maxX=dataX.max(axis = 1, skipna = True)
    maxY=dataY.max(axis = 1, skipna = True)
    maxZ=dataZ.max(axis = 1, skipna = True)
    minX=dataX.min(axis = 1, skipna = True)
    minY=dataY.min(axis = 1, skipna = True)
    minZ=dataZ.min(axis = 1, skipna = True)
    noofmarkers=data.isnull().sum(axis=1)
    
    features=pd.concat([noofmarkers,meanX,meanY,meanZ,stdX,stdY,stdZ,maxX,maxY,maxZ,minX,minY,minZ],axis=1)
    labels=data['Class']
    groups = np.array(data['User'])
    if(train==1):    
        return features,labels,groups
    else:
        return features,labels
    
def SVM_classifier(features,labels,groups,features_test,labelstest):
    
    mean_value=np.mean(features,axis=0)
    
    standard_deviation=np.std(features,axis=0)
    
    features=(features-mean_value)/standard_deviation
    
    features_test=(features_test-mean_value)/standard_deviation
    
    cv_method=LeaveOneGroupOut()
    groups = np.array(data['User'])
    clf=SVC()
    
    tuned_parameters = [{'kernel': ['rbf'],
                         'C': np.logspace(-3,0,30)}]
    
    
    clf = GridSearchCV(estimator=clf,
                       param_grid=tuned_parameters,
                       cv=cv_method,
                       scoring='accuracy')
    
    clf.fit(features, labels,groups)
    
    print("Best parameters")
    print(clf.best_params_)
    print("Best cross validation accuracy")
    print(clf.best_score_)
    means = clf.cv_results_['mean_test_score']
    y_pred =clf.predict(features_test)
    print("Testing Accuracy")
    print(accuracy_score(labelstest, y_pred))
    print("Training Accuracy")
    print(accuracy_score(labels, clf.predict(features)))
    
def KNN_classifier(features,labels,groups,features_test,labelstest):
    
    features=PowerTransformer().fit_transform(features)
    features_test=PowerTransformer().fit_transform(features_test)
    
    poly = PolynomialFeatures(2)
    poly.fit(features)
    extendedfeature=poly.transform(features)
    
    poly = PolynomialFeatures(2)
    poly.fit(features_test)
    extendedfeaturetest=poly.transform(features_test)
    
    lda = LinearDiscriminantAnalysis(n_components=4)
    
    X_r2 = lda.fit(extendedfeature, labels).transform(extendedfeature)
    
    X_r3=lda.transform(extendedfeaturetest)
    
    params_KNN = {'n_neighbors': range(1,30)}
    
    cv_method=LeaveOneGroupOut()
    groups = np.array(data['User'])
    
    gs_KNN = GridSearchCV(estimator=KNeighborsClassifier(), 
                          param_grid=params_KNN, 
                          cv=cv_method,
                          verbose=1,  # verbose: the higher, the more messages
                          scoring='accuracy')
    
    
    features=X_r2
    features_test=X_r3
    
    gs_KNN.fit(features,labels,groups);
    print("Training")
    print("Best parameters")
    print(gs_KNN.best_params_)
    print("Best Cross Validation Accuracy")
    print(gs_KNN.best_score_)
    print("Testing Accuracy")
    y_pred=gs_KNN.predict(features_test)
    print(accuracy_score(labelstest, y_pred))
    print("Training Accuracy")
    print(accuracy_score(labels, gs_KNN.predict(features)))
    print(gs_KNN)
    
def Naive_Bayes_Classifier(features,labels,groups,features_test,labelstest):
    features=PowerTransformer().fit_transform(features)
    features_test=PowerTransformer().fit_transform(features_test)
    
    #----------------------------EXTENDED FEATURE SPACE AND LDA-------------------------------------------#
    poly = PolynomialFeatures(2)
    poly.fit(features)
    extendedfeature=poly.transform(features)
    
    poly = PolynomialFeatures(2)
    poly.fit(features_test)
    extendedfeaturetest=poly.transform(features_test)
    
    extendedfeature=PowerTransformer().fit_transform(extendedfeature)
    extendedfeaturetest=PowerTransformer().fit_transform(extendedfeaturetest)
    
    lda = LinearDiscriminantAnalysis(n_components=5)
    
    X_r2 = lda.fit(extendedfeature, labels).transform(extendedfeature)
    
    X_r3=lda.transform(extendedfeaturetest)
    
    #---------------------------NAIVE BAYES CLASSIFIER---------------------------------------------------------#
    
    params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
    
    cv_method=LeaveOneGroupOut()
    groups = np.array(data['User'])
    
    nb_classifier = GaussianNB()
    
    gs_NB = GridSearchCV(estimator=nb_classifier, 
                         param_grid=params_NB, 
                         cv=cv_method,
                         verbose=1, 
                         scoring='accuracy')
    
    
    
    gs_NB.fit(X_r2,labels,groups);
    
    print(gs_NB.best_params_)
    print("Best Cross Validation Accuracy")
    print(round(gs_NB.best_score_ , 4))
    y_pred=gs_NB.predict(X_r3)
    print("Testing Accuracy")
    print(round(accuracy_score(labelstest, y_pred),4))
    print("Training Accuracy")
    print(round(accuracy_score(labels,gs_NB.predict(X_r2) ),4))
    print(gs_NB)

def Randomforest_Classifier(features,labels,groups,features_test,labelstest):
    features=PowerTransformer().fit_transform(features)
    features_test=PowerTransformer().fit_transform(features_test)
    
    rfc=RandomForestClassifier()
    rfc.fit(features,labels)
    print("Training Accuracy {}".format(accuracy_score(labels, rfc.predict(features))*100))
    print("Testing Accuracy {}".format(accuracy_score(labelstest, rfc.predict(features_test))*100))
    
    
    param_grid = { 
        'n_estimators': [500,600,700],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth':[40,60,80]
    }
    rfc=RandomForestClassifier()
    
    cv_method=LeaveOneGroupOut()
    groups = np.array(data['User'])
    
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= cv_method)
    CV_rfc.fit(features, labels,groups)
    print(CV_rfc.best_params_)
    
    print("Best parameters")
    print(CV_rfc.best_params_)
    print("Best cross validation accuracy")
    print(CV_rfc.best_score_)
    y_pred =CV_rfc.predict(features_test)
    print("Testing Accuracy")
    print(accuracy_score(labelstest, y_pred))
    print("Training Accuracy")
    print(accuracy_score(labels, CV_rfc.predict(features)))
    
def LogisticCLassifer(features,labels,groups,features_test,labelstest):
    mean_value=np.mean(features,axis=0)

    standard_deviation=np.std(features,axis=0)
    
    features=(features-mean_value)/standard_deviation
    
    #mean_valuetest=np.mean(features_test,axis=0)
    
    #standard_deviationtest=np.std(features_test,axis=0)
    
    
    features_test=(features_test-mean_value)/standard_deviation
    
    
    
    
    cv_method=LeaveOneGroupOut()
    groups = np.array(data['User'])
    
    clf=LogisticRegressionCV(random_state=0).fit(features,labels,groups)
    y_pred=clf.predict(features_test)
    print("testing accuracy")
    print(accuracy_score(labelstest,y_pred))
    print("training accuracy")
    print(accuracy_score(labels,clf.predict(features)))

if __name__ == '__main__':
    
    #Reading the files
    data=pd.read_csv('D:/mpr/project/hand_postures/D_train.csv')
    datatest=pd.read_csv('D:/mpr/project/hand_postures/D_test.csv')
    train=1
    features,labels,groups=feature_extraction(data,train)
    train=0
    features_test,labelstest=feature_extraction(datatest,train)

    print("SVM Classification")
    SVM_classifier(features,labels,groups,features_test,labelstest)
    print("Naive Bayes Classification")
    Naive_Bayes_Classifier(features,labels,groups,features_test,labelstest)
    print("KNN Classification")
    KNN_classifier(features,labels,groups,features_test,labelstest)
    print("Random Forest Classfication")
    Randomforest_Classifier(features,labels,groups,features_test,labelstest)
    print("Logistic regression")
    LogisticCLassifer(features,labels,groups,features_test,labelstest)
    
    
    

    
