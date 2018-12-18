# Goal: Visualize protein distribution in two dimensions to rationalize discrepant predictions
# How: Nonlinear dimensionality reduction using t-SNE and coloring according
# to prediction class and functional subclass
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
# get data for which predictions of SURFY and SURFY2 exist
random.seed(42)
df = pd.read_csv('pred_unlabeled_proteins.csv')
surfy = pd.read_excel('surfy.xlsx')
surfy = surfy[surfy['Surfaceome Label Source'] == 'machine learning']
surfy.Surfaceome_Label = surfy.Surfaceome_Label.replace('nonsurface',0)
surfy.Surfaceome_Label = surfy.Surfaceome_Label.replace('surface',1)
surfy.dropna(subset=['Surfaceome_Label'], inplace=True)
names = surfy.UniProt_name
df = df[df['name'].isin(names)]
df2 = df.drop(['accession','name','surface','score'],axis=1)

# dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40,random_state=42,
            n_iter=1500)
tsne_results = tsne.fit_transform(df2)
df['x-tsne'] = tsne_results[:,0]
df['y-tsne'] = tsne_results[:,1]
surfy.rename(columns={'UniProt_name':'name'},inplace=True)
df3 = pd.merge(df,surfy,on='name')

# plot t-SNE and color proteins according to predicted class label
sns.set(style="white", color_codes=True)
plt.figure()
plt.scatter(df3.loc[(df3['surface'] == 0) & (df3['Surfaceome_Label']==0), ['x-tsne']],
                   df3.loc[(df3['surface'] == 0) & (df3['Surfaceome_Label']==0),['y-tsne']],
                   marker='o', color='g',
                   linewidth='1', alpha=0.6, label='Intra_shared')
plt.scatter(df3.loc[(df3['surface'] == 1)&(df3['Surfaceome_Label']==1), ['x-tsne']],
                   df3.loc[(df3['surface'] == 1)&(df3['Surfaceome_Label']==1), ['y-tsne']],
                   marker='o', color='orange',
                   linewidth='1', alpha=0.6, label='Surface_shared')
plt.scatter(df3.loc[(df3['surface'] == 0) & (df3['Surfaceome_Label']==1), ['x-tsne']],
                   df3.loc[(df3['surface'] == 0) & (df3['Surfaceome_Label']==1),['y-tsne']],
                   marker='o', color='b',
                   linewidth='1', alpha=0.6, label='Intra_diff')
plt.scatter(df3.loc[(df3['surface'] == 1)&(df3['Surfaceome_Label']==0), ['x-tsne']],
                   df3.loc[(df3['surface'] == 1)&(df3['Surfaceome_Label']==0), ['y-tsne']],
                   marker='o', color='k',
                   linewidth='1', alpha=0.6, label='Surface_diff')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('T-SNE labeled by predicted classes')
plt.legend(loc='best')
plt.show()

# plot t-SNE of surface proteins and color according to functional class
func = pd.read_excel('surface_characteristics.xlsx')
func.rename(columns={'UniProt_name':'name'},inplace=True)
func = func[func['name'].isin(names)]
df3 = pd.merge(df3,func,on='name')
plt.figure()
colors = ['g','r','b','y','k']
names = list(df3.MembranomeAlmenMainClass.unique())
for i in range(5):
    plt.scatter(df3.loc[df3['MembranomeAlmenMainClass'] == names[i], ['x-tsne']],
                   df3.loc[df3['MembranomeAlmenMainClass'] == names[i],['y-tsne']],
                   marker='o', color=colors[i],
                   linewidth='1', alpha=0.6, label=names[i])
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('T-SNE labeled by functional class')
plt.legend(loc='best')
plt.show()

# plot t-SNE of surface proteins with prevalent function and color according
# to functional subclass
from collections import Counter
countr = Counter(df3.MembranomeAlmenSubClass.values.tolist())
names = countr.most_common(11)
# delete 'nan' functional subclass
del(names[5])
plt.figure()
colors = ['g','maroon','b','y','k','c','m','orange','darkgreen','saddlebrown']
for i in range(10):
    plt.scatter(df3.loc[df3['MembranomeAlmenSubClass'] == names[i][0], ['x-tsne']],
                   df3.loc[df3['MembranomeAlmenSubClass'] == names[i][0],['y-tsne']],
                   marker='o', color=colors[i],
                   linewidth='1', alpha=0.6, label=names[i][0])
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('T-SNE labeled by functional sub-class')
plt.legend(loc='best',prop={'size': 6})
plt.show()
