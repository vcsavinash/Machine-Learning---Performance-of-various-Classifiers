import sys
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  cross_val_predict
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment=None

data = pd.read_excel(sys.argv[1],header=0,skiprows=[1])
data.replace(to_replace="[?]",value=np.nan,regex=True,inplace=True)
data = data.dropna()
attributes = ['X1','X2', 'X3','X4','X5','X6','X7','X8','X9']
temp = data.loc[:,'X1':'X9']
scaledData = preprocessing.StandardScaler()
scaledFinal = scaledData.fit(temp)
dataStandardized = scaledFinal.transform(temp)
standardData = pd.DataFrame(dataStandardized, columns=attributes)

decisionTree = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='sqrt')
perceptron = Perceptron(penalty = 'l1',alpha=0.001,max_iter=200,shuffle=True)
neuralNet = MLPClassifier(hidden_layer_sizes=(8,6),activation='tanh',alpha=0.005,learning_rate='adaptive',max_iter=200,learning_rate_init=0.001,solver='lbfgs')
deepLearning = MLPClassifier(hidden_layer_sizes=(20,16,14,10,9,8,7,6,5,4,2),activation='tanh',solver='lbfgs',alpha=0.005,learning_rate='adaptive',max_iter=60,learning_rate_init=0.001)
SVM = SVC(C=1.0, kernel='rbf',max_iter=70)
gaussianNaiveBayes = GaussianNB()
logisticRegression = LogisticRegression(penalty='l1',max_iter=100,solver='liblinear',C=2)
kNN = KNeighborsClassifier(n_neighbors=20, algorithm='kd_tree')
bagging = BaggingClassifier(base_estimator=neuralNet,n_estimators=10,bootstrap=True)
randomForest = RandomForestClassifier(n_estimators=50,criterion='gini',max_features='auto')
adaBoost = AdaBoostClassifier(base_estimator=SVM,n_estimators=80,learning_rate=0.001,algorithm='SAMME')
gradientBoost = GradientBoostingClassifier(loss= 'deviance', learning_rate=0.05, n_estimators=100, presort='auto')
kf = KFold(n_splits=10,shuffle=True)

classifiers ={}
classifiers['Decision Tree'] = decisionTree
classifiers['Perceptron'] = perceptron
classifiers['Neural Network'] = neuralNet
classifiers['Deep Learning'] = deepLearning
classifiers['SVM'] = SVM
classifiers['Naive Bayes'] = gaussianNaiveBayes
classifiers['Logistic Regression'] = logisticRegression
classifiers['kNN'] = kNN
classifiers['Bagging'] = bagging
classifiers['Random Forest'] = randomForest
classifiers['ADA Boost'] = adaBoost
classifiers['Gradient Boost'] = gradientBoost

for classifier in classifiers.keys():
	method = classifiers[classifier]
	output = cross_val_predict(method,standardData.loc[:,'X1':'X9'],data.loc[:,'Y'],cv=kf)
	print (classifier)
	print ("Accuracy:",accuracy_score(data.loc[:,'Y'],output))
	print ("F1 Score:",f1_score(data.loc[:,'Y'],output, labels=[2, 4], average=None))