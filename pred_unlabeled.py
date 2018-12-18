# Goal: Get the labels for all the other proteins
# How: Train StackingClassifier on Train+Validation+Test data and predict unlabeled data
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
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
from sklearn.metrics import roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
random.seed(42)
# getting all the data and merging train/val/test data into train data
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')
train = pd.concat([train,val],axis=0)
train = pd.concat([train,test],axis=0)
y = train.surface
X = train.drop(['surface'],axis=1)
X_acc = X.accession
X_name = X.name
X = X.drop(['accession','name'],axis=1)

# getting & preparing the unlabeled data points
df = pd.read_excel('data.xlsx')
Xt = df.drop(df[df['name'].isin(X_name)].index)
Xt_acc = Xt.accession
Xt_name = Xt.name
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

# Building and running the StackingClassifier on the unlabeled data
from mlxtend.classifier import StackingCVClassifier
sclf=StackingCVClassifier(classifiers=[rf,lr,gb,et,gnb,svc,knn,xgb,ada,mlp,lda,qda],
                          use_features_in_secondary=True,
                          use_probas=True,
                          store_train_meta_features=True,
                        meta_classifier=eclf)
sclf.fit(X.values,y.values)

# get feature importances of voting classifier estimators
est=sclf.meta_clf_
RF,LR,GB=est.estimators_
new = sclf.train_meta_features_.tolist()
new =pd.DataFrame(data=sclf.train_meta_features_.tolist(),
		       columns=['rf_0','rf_1','lr_0','lr_1','gb_0','gb_1',
                                'et_0','et_1','gnb_0','gnb_1','svc_0',
                                'svc_1','knn_0','knn_1','xgb_0',
                                'xgb_1','ada_0','ada_1','mlp_0',
                                'mlp_1','lda_0','lda_1','qda_0',
                                'qda_1'])
X.reset_index(inplace=True, drop=True)
X = pd.concat([X,new],axis=1)
names=X.columns.values.tolist()
rf_fi = RF.feature_importances_.tolist()
lr_fi = LR.coef_.tolist()[0]
gb_fi = GB.feature_importances_.tolist()
fi = np.column_stack((names,rf_fi,lr_fi,gb_fi))
fi = pd.DataFrame(data=fi,columns=['feature','RF','LR','GB'])
fi.to_csv('feature_importances.csv',encoding='utf-8',index=False)

# get predictions and prediction scores for the unlabeled proteins
pred=sclf.predict(Xt.values)
pred_proba=sclf.predict_proba(Xt.values)
Xt['surface'] = pred
Xt['score']=pred_proba[:,1]
Xt = pd.concat([Xt_acc, Xt_name, Xt],axis=1)
Xt.to_csv('pred_unlabeled_proteins.csv',encoding='utf-8',index=False)
