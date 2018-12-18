# Goal: Plot Feature Importances
# How: Take the three classifiers making up the meta-estimator VotingClassifier
# and plot their sorted feature importances
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
random.seed = 42
# get the feature importances of the votingClassifier estimators (from stacking)
df=pd.read_csv('feature_importances.csv')


# plot 10 most important features for Random Forest
sns.set(style="whitegrid")
rf = df.sort_values(by=['RF'],ascending=False)
rf = rf.iloc[:10,:]
ax=sns.barplot(x='feature',y='RF',data=rf,
               palette="Blues_d")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('Random Forest')
ax.set_xlabel('Feature')
ax.set_ylabel('Relative Importance')
plt.savefig("rf_feature_importance.png",bbox_inches='tight')
plt.clf()

# plot 10 most important features (5 positive, 5 negative) for Logistic Regression
lr = df.sort_values(by=['LR'],ascending=False)
lr1 = lr.iloc[:5,:]
lr2=lr.iloc[-5:,:]
lr = pd.concat([lr1,lr2],axis=0)
ax=sns.barplot(x='feature',y='LR',data=lr,
               palette="Blues_d")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('Logistic Regression')
ax.set_xlabel('Feature')
ax.set_ylabel('Relative Importance')
plt.savefig("lr_feature_importance.png",bbox_inches='tight')
plt.clf()

# plot 10 most important features for Gradient Boosting
gb = df.sort_values(by=['GB'],ascending=False)
gb = gb.iloc[:10,:]
ax=sns.barplot(x='feature',y='GB',data=gb,
               palette="Blues_d")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('Gradient Boosting')
ax.set_xlabel('Feature')
ax.set_ylabel('Relative Importance')
plt.savefig("gb_feature_importance.png",bbox_inches='tight')
plt.clf()
