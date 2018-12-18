# Goal: Test the classifier with the test data
# How: Train StackingClassifier on train+validation data and predict test data
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
random.seed(42)
# getting all the data and merging train & validation data into new train data
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')
train = pd.concat([train,val],axis=0)
y = train.surface
X = train.drop(['surface'],axis=1)
X = X.drop(['accession','name'],axis=1)
yt = test.surface
Xt = test.drop(['surface'],axis=1)
Xt = Xt.drop(['accession','name'],axis=1)

# optimized base classifiers
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

# Building and running the StackingClassifier on the test data
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
plt.savefig('ROC_curve_test.png',bbox_inches='tight')
plt.clf()
perf = pd.read_csv('performance_estimates.csv')
metrics=[]
metrics.append(accuracy_score(yt,pred))
metrics.append(precision_score(yt,pred))
metrics.append(recall_score(yt,pred))
metrics.append(roc_auc_score(yt,pred))
metrics = np.column_stack((cmetrics,metrics))
metrics = pd.DataFrame(data=metrics,columns=['Crossval_Testset','Testset'])
perf = pd.concat([perf,metrics],axis=1)
perf.to_csv('performance_estimates.csv',encoding='utf-8',index=False)
print(confusion_matrix(yt,pred))
