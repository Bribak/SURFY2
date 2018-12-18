# Goal: Get features importances from relevant base estimators
# How: Train XGBClassifier and GradientBoosting on train+val+test data and get
# as well as plot their feature importances
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
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

# optimized base classifiers
xgb = XGBClassifier(subsample=0.8,
                                                n_estimators=200,
                                                random_state=42)
gb = GradientBoostingClassifier(max_depth=4,
                                                             loss='exponential',
                                                             n_estimators=200,
                                                         random_state=42)

# fit base estimators and get their feature importances into a dataframe
xgb.fit(X,y)
xgb_fi = xgb.feature_importances_.tolist()
gb.fit(X,y)
gb_fi = gb.feature_importances_.tolist()
names=X.columns.values.tolist()
fi = np.column_stack((names,xgb_fi,gb_fi))
fi = pd.DataFrame(data=fi,columns=['feature','XGB','GB'])
sns.set(style="whitegrid")
fi['XGB']=fi['XGB'].astype('float')
fi['GB']=fi['GB'].astype('float')

# plot 10 most important features for XGBClassifier
xgb = fi.sort_values(by=['XGB'],ascending=False)
xgb = xgb.iloc[:10,:]
ax=sns.barplot(x='feature',y='XGB',data=xgb,
               palette='Blues_d')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('XGBClassifier base estimator')
ax.set_xlabel('Feature')
ax.set_ylabel('Relative Importance')
plt.savefig("xgb_base_feature_importance.png",bbox_inches='tight')
plt.clf()

# plot 10 most important features for GradientBoosting
gb = fi.sort_values(by=['GB'],ascending=False)
gb = gb.iloc[:10,:]
ax=sns.barplot(x='feature',y='GB',data=gb,
               palette='Blues_d')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('GradientBoosting base estimator')
ax.set_xlabel('Feature')
ax.set_ylabel('Relative Importance')
plt.savefig("gb_base_feature_importance.png",bbox_inches='tight')
plt.clf()
