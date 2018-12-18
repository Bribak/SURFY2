# Goal: Assess the metrics of SURFY
# How: Fit the RandomForest classifier of SURFY on the datasets and report metrics
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
random.seed(42)
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
y = train.surface
X = train.drop(['surface'],axis=1)
X = X.drop(['accession','name'],axis=1)
yt = val.surface
Xt = val.drop(['surface'],axis=1)
Xt = Xt.drop(['accession','name'],axis=1)

# random forest classifier
rf = RandomForestClassifier(n_estimators=501,criterion='gini',random_state=42)

# evaluate through cross-validation
cmetrics=[]
cmetrics.append(cross_val_score(rf,X,y,cv=5,scoring='accuracy').mean())
cmetrics.append(cross_val_score(rf,X,y,cv=5,scoring='precision').mean())
cmetrics.append(cross_val_score(rf,X,y,cv=5,scoring='recall').mean())
cmetrics.append(cross_val_score(rf,X,y,cv=5,scoring='roc_auc').mean())
rf.fit(X,y)
pred=rf.predict(Xt)

# plotting ROC-Curve
pred_proba=rf.predict_proba(Xt)[:,1]
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
plt.savefig('ROC_curve_val_SURFY.png',bbox_inches='tight')
plt.clf()

# evaluate on the validation set
metrics=[]
metrics.append(accuracy_score(yt,pred))
metrics.append(precision_score(yt,pred))
metrics.append(recall_score(yt,pred))
metrics.append(roc_auc_score(yt,pred))
names=['Accuracy','Precision','Recall','Area under ROC curve']

# evaluate cross-validation after training on (training+validation) data
train = pd.concat([train,val],axis=0)
y = train.surface
X = train.drop(['surface','accession','name'],axis=1)
test = pd.read_csv('test.csv')
yt = test.surface
Xt = test.drop(['surface','accession','name'],axis=1)
ctmetrics=[]
ctmetrics.append(cross_val_score(rf,X,y,cv=5,scoring='accuracy').mean())
ctmetrics.append(cross_val_score(rf,X,y,cv=5,scoring='precision').mean())
ctmetrics.append(cross_val_score(rf,X,y,cv=5,scoring='recall').mean())
ctmetrics.append(cross_val_score(rf,X,y,cv=5,scoring='roc_auc').mean())
rf.fit(X,y)
pred=rf.predict(Xt)

# plotting ROC-Curve
pred_proba=rf.predict_proba(Xt)[:,1]
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
plt.savefig('ROC_curve_val_SURFY.png',bbox_inches='tight')
plt.clf()

# evaluate on the test set
tmetrics=[]
tmetrics.append(accuracy_score(yt,pred))
tmetrics.append(precision_score(yt,pred))
tmetrics.append(recall_score(yt,pred))
tmetrics.append(roc_auc_score(yt,pred))
table = np.column_stack((names,cmetrics,metrics,ctmetrics,tmetrics))
table=pd.DataFrame(data=table,columns=['Metrics','Crossval_Validation',
                                       'Validation','Crossval_Testset',
                                       'Testset'])
table.to_csv('performance_estimates_SURFY.csv',encoding='utf-8',index=False)
print(table)
print(confusion_matrix(yt,pred))
