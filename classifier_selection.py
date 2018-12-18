# Goal: get all the crossvalidation accuracies in one table
# How: perform 5fold crossvalidation with all classifiers and only the train data
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
random.seed(42)
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
y = train.surface
X = train.drop(['surface'],axis=1)
X = X.drop(['accession','name'],axis=1)
yt = val.surface
Xt = val.drop(['surface'],axis=1)
Xt = Xt.drop(['accession','name'],axis=1)

# optimized base estimators
rf = RandomForestClassifier(min_samples_split=8,
                                                         n_estimators=200,
                                                         max_depth=20,
                                                         criterion='gini',
                                                         random_state=42)
lr = LogisticRegression(class_weight='balanced',max_iter=400,
                                                     solver='newton-cg',
                                                         random_state=42)
gb = GradientBoostingClassifier(max_depth=4,
                                                             loss='exponential',
                                                             n_estimators=200,
                                                         random_state=42)
et = ExtraTreesClassifier(max_features=26,
                                                       n_estimators=220,
                                                       max_depth=24,
                                                       criterion='gini',
                                                         random_state=42)
gnb = GaussianNB(
                                                         )
svc=SVC(max_iter=400,
                                      gamma='scale',
                                      probability=True,
                                    random_state=42)
knn = KNeighborsClassifier(p=1,
                                                       n_neighbors=8,
                                                       algorithm='ball_tree',
                                                       weights='distance')
xgb = XGBClassifier(subsample=0.8,
                                                n_estimators=200,
                                                random_state=42)
ada = AdaBoostClassifier(n_estimators=160,
                                                random_state=42)
mlp = MLPClassifier(alpha=0.00005,
                                                activation='tanh',
                                                random_state=42)
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

# optimized votingClassifier
eclf = VotingClassifier(estimators=[('rf',rf),('lr',lr),('gb',gb)],voting='soft',
                        weights=[3,2,3])
from mlxtend.classifier import StackingCVClassifier
print('5-fold cross validation:\n')
names =['RandomForest','RandomForest_FE',
        'LogisticRegression','LogisticRegression_FE',
        'GradientBoosting','GradientBoosting_FE',
        'ExtraTrees','ExtraTrees_FE','GaussianNB',
        'GaussianNB_FE','SVC','SVC_FE','KNN','KNN_FE',
        'XGBClassifier','XGBClassifier_FE',
        'AdaBoost','AdaBoost_FE','MLP','MLP_FE',
        'LDA','LDA_FE','QDA','QDA_FE',
        'VotingClassifier','VotingClassifier_FE']
scores_mean=[]
scores_std=[]
for clf in [rf,lr,gb,et,gnb,svc,knn,xgb,
                       ada,mlp,lda,qda,eclf]:
    sclf=StackingCVClassifier(classifiers=[rf,lr,gb,et,gnb,svc,knn,xgb,ada,mlp,lda,qda],
                          use_features_in_secondary=True,
                              use_probas=True,
                        meta_classifier=clf)
    score1 =cross_val_score(clf, X.values, y.values, 
                                              cv=5, scoring='accuracy')
    score2 = cross_val_score(sclf,X.values,y.values,
                                  cv=5,scoring='accuracy')
    scores_mean.append(score1.mean())
    scores_mean.append(score2.mean())
    scores_std.append(score1.std())
    scores_std.append(score2.std())
out = np.column_stack((names,scores_mean,scores_std))
out = pd.DataFrame(data=out,columns=['Classifier','Crossvalidation Accuracy Mean',
                                     'Standard Deviation'])
out.to_csv('classifier_selection.csv',encoding='utf-8',index=False)
