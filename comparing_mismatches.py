# Goal: analyze why some samples were classified differently by SURFY2
# How: select features which are most different between shared/different and interpret
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
import statsmodels.stats.multitest as smm
random.seed(42)
df = pd.read_csv('pred_unlabeled_proteins.csv')
mismatches = pd.read_csv('mismatches.csv')

# separate data features into shared classifications and different classifications
mis = df[df['name'].isin(mismatches.name)]
rest = df[~df['name'].isin(mismatches.name)]
shared = rest.groupby('surface').mean()
different = mis.groupby('surface').mean()
shared.to_csv('shared_pred_features.csv',encoding='utf-8',index=False)
different.to_csv('different_pred_features.csv',encoding='utf-8',index=False)

# compare means of the two data groups and select highest differences
names=[]
differenceS=[]
differenceD=[]
delta=[]
cols = shared.columns.values
for column in range(shared.shape[1]):
    names.append(cols[column])
    differenceS.append(shared.iloc[1,column]-shared.iloc[0,column])
    differenceD.append(different.iloc[1,column]-different.iloc[0,column])
    delta.append(abs((shared.iloc[1,column]-shared.iloc[0,column])-
                     (different.iloc[1,column]-different.iloc[0,column])))
result = np.column_stack((names,differenceS,differenceD,delta))
result=pd.DataFrame(data=result,columns=['feature','shared_diff','diff_diff',
                     'delta'])
result.delta=result.delta.astype(float)
result = result[result['delta']>0.5]
mis2 = mis.loc[:,result.feature.values.tolist()]
mis2['surface']=mis['surface']
rest2 = rest.loc[:,result.feature.values.tolist()]
rest2['surface']=rest['surface']

# compare means of features which are most different between shared/different
shared2=rest2.groupby('surface').mean()
s_std=rest2.groupby('surface').agg(np.std)
N_shared_0 = rest2.surface.value_counts()[0]
N_shared_1 = rest2.surface.value_counts()[1]
different2=mis2.groupby('surface').mean()
d_std=mis2.groupby('surface').agg(np.std)
N_diff_0 = mis2.surface.value_counts()[0]
N_diff_1 = mis2.surface.value_counts()[1]
surface=['0_shared','1_shared','0_diff','1_diff']
surface=pd.DataFrame(data=surface,columns=['surface'])
data2 = pd.concat([shared2,different2],axis=0)
data2=data2.reset_index(drop=True)
data = pd.concat([data2,surface],axis=1)
stds = pd.concat([s_std,d_std],axis=0)

# check significant differences between class means
names= list(data2)
pvalues_shared = []
pvalues_different =[]
for name in names:
    tstat, pvalue_shared=ttest_ind_from_stats(data[name].iloc[0],stds[name].iloc[0],N_shared_0,
                                              data[name].iloc[1],stds[name].iloc[1],N_shared_1,
                                              equal_var=False)
    pvalues_shared.append(pvalue_shared)
    tstat, pvalue_diff=ttest_ind_from_stats(data[name].iloc[2],stds[name].iloc[2],N_diff_0,
                                              data[name].iloc[3],stds[name].iloc[3],N_diff_1,
                                            equal_var=False)
    pvalues_different.append(pvalue_diff)
pval_corr_shared = smm.multipletests(np.asarray(pvalues_shared), alpha=0.05,
                                              method='hs')[1]
pval_corr_diff= smm.multipletests(np.asarray(pvalues_different), alpha=0.05,
                                              method='hs')[1]
p_values=np.column_stack((names,pval_corr_shared,pval_corr_diff))
p_values=pd.DataFrame(data=p_values,columns=['name','pvalue_shared','pvalue_different'])
p_values.to_csv('pval_mismatches2.csv',encoding='utf-8',index=False)
shared2.to_csv('shared2_pred_features.csv',encoding='utf-8',index=False)
different2.to_csv('different2_pred_features.csv',encoding='utf-8',index=False)

# plot 9 representative features for shared/different samples
style.use('ggplot')
sns.set_context('paper')
f,axes=plt.subplots(3,3)
first =sns.barplot(y='ncd_count',x='surface',data=data,ax=axes[0,0])
first.errorbar(data.index, data['ncd_count'], yerr=stds['ncd_count'], fmt='ko',alpha=0.6)
second=sns.barplot(y='NxST_absolute',x='surface',data=data,ax=axes[0,1])
second.errorbar(data.index, data['NxST_absolute'], yerr=stds['NxST_absolute'], fmt='ko',alpha=0.6)
third=sns.barplot(y='tmd_length_average',x='surface',data=data,ax=axes[0,2])
third.errorbar(data.index, data['tmd_length_average'], yerr=stds['tmd_length_average'], fmt='ko',alpha=0.6)
fourth=sns.barplot(y='ncd_cys_absolute',x='surface',data=data,ax=axes[1,0])
fourth.errorbar(data.index, data['ncd_cys_absolute'], yerr=stds['ncd_cys_absolute'], fmt='ko',alpha=0.6)
fifth=sns.barplot(y='ncd_cgly_absolute',x='surface',data=data,ax=axes[1,1])
fifth.errorbar(data.index, data['ncd_cgly_absolute'], yerr=stds['ncd_cgly_absolute'], fmt='ko',alpha=0.6)
sixth=sns.barplot(y='tmd_count',x='surface',data=data,ax=axes[1,2])
sixth.errorbar(data.index, data['tmd_count'], yerr=stds['tmd_count'], fmt='ko',alpha=0.6)
seventh=sns.barplot(y='cytoplasmic+count',x='surface',data=data,ax=axes[2,0])
seventh.errorbar(data.index, data['cytoplasmic+count'], yerr=stds['cytoplasmic+count'], fmt='ko',alpha=0.6)
eight=sns.barplot(y='signalpeptide+count',x='surface',data=data,ax=axes[2,1])
eight.errorbar(data.index, data['signalpeptide+count'], yerr=stds['signalpeptide+count'], fmt='ko',alpha=0.6)
ninth=sns.barplot(y='transmembrane+length',x='surface',data=data,ax=axes[2,2])
ninth.errorbar(data.index, data['transmembrane+length'], yerr=stds['transmembrane+length'], fmt='ko',alpha=0.6)
plt.tight_layout(pad=0.3)
f.savefig('mismatch_features.png',dpi=f.dpi)
plt.show()
plt.clf()
