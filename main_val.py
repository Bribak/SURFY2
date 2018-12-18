# Goal: Build a Classifier
# How: Optimize base estimators, optimize a soft VotingClassifier with optimal weights,
# build a StackingClassifier with base estimators & VotingClassifier
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
random.seed(42)
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
y = train.surface
X = train.drop(['surface'],axis=1)
X = X.drop(['accession','name'],axis=1)
yt = val.surface
Xt = val.drop(['surface'],axis=1)
Xt = Xt.drop(['accession','name'],axis=1)

### Code segment to optimize base classifiers with cross-validation
##param_grid = {'nu':[0.4,0.5,0.6]}
##gsearch1 = GridSearchCV(estimator=NuSVC(probability=True,
##                                        gamma='scale',
##                                        max_iter=200,
##                                        random_state=42),
##                        param_grid=param_grid, scoring='balanced_accuracy',
##                        iid=False,cv=5)
##gsearch1.fit(X,y)
##print("Grid scores on development set:")
##means = gsearch1.cv_results_['mean_test_score']
##stds = gsearch1.cv_results_['std_test_score']
##for mean, std, params in zip(means, stds, gsearch1.cv_results_['params']):
##    print("%0.3f (+/-%0.03f) for %r"
##          % (mean, std * 2, params))
##print(gsearch1.best_params_, gsearch1.best_score_)
##pred=gsearch1.best_estimator_.predict(Xt)
##print(confusion_matrix(yt,pred))

# Optimized base classifiers
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

### testing optimized votingClassifier
##for clf, label in zip([rf,lr,gb,eclf],
##                      ['Random Forest','Logistic Regression',
##                       'Gradient Boosting','Ensemble']):
##
##    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
##    print("Accuracy: %0.5f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))
##eclf.fit(X,y)
##pred=eclf.predict(Xt)
##print(confusion_matrix(yt,pred))
### building the best votingClassifier from the optimized base classifiers by greedy selection
##classifiers = [('et',et),('gnb',gnb),('svc',svc),
##                                    ('knn',knn),('xgb',xgb),('ada',ada),
##                                    ('mlp',mlp)]
##for est in classifiers:
##    eclf = VotingClassifier(estimators=[('gb',gb),('rf',rf),
##                                        ('lr',lr),est]
##                            ,voting='soft',weights=[3,3,2,1])
##    scores = cross_val_score(eclf,X,y,cv=5,scoring='balanced_accuracy')
##    print(est)
##    print(scores.mean())

### Voting weight optimization of optimal votingClassifier
##dweights = pd.DataFrame(columns=('w1', 'w2','w3', 'mean', 'std'))
##i=0
##for w1 in range(1,4):
##    for w2 in range(1,4):
##        for w3 in range(1,4):
##            if len(set((w1,w2,w3)))==1:
##                continue
##            eclf = VotingClassifier(estimators=[('gb',gb),('rf',rf),
##                                                ('lr',lr)],
##                                    voting='soft',weights=[w1,w2,w3])
##            scores = cross_val_score(eclf,X,y,cv=5,scoring='balanced_accuracy')
##            dweights.loc[i]=[w1,w2,w3,scores.mean(),scores.std()]
##            i+=1
##dweights = dweights.sort_values(['mean','std'],ascending=False)
##print(dweights)
##eclf = VotingClassifier(estimators=[('gb',gb),('rf',rf),('lr',lr)],
##                                voting='soft',weights=[3,3,2])
##scores = cross_val_score(eclf,X,y,cv=10,scoring='balanced_accuracy')
##print(scores.mean())
##print(scores.std())
##eclf.fit(X,y)
##pred=eclf.predict(Xt)
##print(confusion_matrix(yt,pred))

### Building and running the StackingClassifier on the validation set
from mlxtend.classifier import StackingCVClassifier
sclf=StackingCVClassifier(classifiers=[rf,lr,gb,et,gnb,svc,knn,xgb,ada,mlp,lda,qda],
                          use_features_in_secondary=True,
                          use_probas=True,
                        meta_classifier=eclf)
cmetrics=[]
cmetrics.append(cross_val_score(sclf,X.values,y.values,cv=5,scoring='accuracy').mean())
cmetrics.append(cross_val_score(sclf,X.values,y.values,cv=5,scoring='precision').mean())
cmetrics.append(cross_val_score(sclf,X.values,y.values,cv=5,scoring='recall').mean())
cmetrics.append(cross_val_score(sclf,X.values,y.values,cv=5,scoring='roc_auc').mean())
sclf.fit(X.values,y.values)
pred=sclf.predict(Xt.values)

# plotting ROC-Curve
pred_proba=sclf.predict_proba(Xt.values)[:,1]
fpr, tpr, threshold = roc_curve(yt, pred_proba)
roc_auc=auc(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('ROC_curve_val.png',bbox_inches='tight')
plt.clf()
metrics=[]
metrics.append(accuracy_score(yt,pred))
metrics.append(precision_score(yt,pred))
metrics.append(recall_score(yt,pred))
metrics.append(roc_auc_score(yt,pred))
names=['Accuracy','Precision','Recall','Area under ROC curve']
table = np.column_stack((names,cmetrics,metrics))
table=pd.DataFrame(data=table,columns=['Metrics','Crossval_Validation','Validation'])
table.to_csv('performance_estimates.csv',encoding='utf-8',index=False)
print(table)
print(confusion_matrix(yt,pred))
