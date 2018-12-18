# Goal: Finding differences in the predictions of SURFY and SURFY2
# How: Find rows in the two datasets which the label columns don't match
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
random.seed(42)
# get SURFY2 labels and the labels of SURFY (+scores)
pred = pd.read_csv('pred_unlabeled_proteins.csv')
surfy = pd.read_excel('surfy.xlsx')
surfy = surfy[surfy['Surfaceome Label Source'] == 'machine learning']
surfy.Surfaceome_Label = surfy.Surfaceome_Label.replace('nonsurface',0)
surfy.Surfaceome_Label = surfy.Surfaceome_Label.replace('surface',1)
surfy.dropna(subset=['Surfaceome_Label'], inplace=True)
names = surfy.UniProt_name
pred = pred[pred['name'].isin(names)]
mismatch=[]
theirs=[]
theirS=[]
mine = []
mineS =[]

# generate a .csv file with all mismatches of SURFY and SURFY2 in terms of prediction and their scores
for name in names:
    if surfy.loc[surfy.UniProt_name==name,['Surfaceome_Label']].values.tolist()!=pred.loc[pred.name==name,['surface']].values.tolist():
        mismatch.append(name)
        theirs.append(surfy.loc[surfy.UniProt_name==name,['Surfaceome_Label']].values.tolist()[0])
        theirS.append(surfy.loc[surfy.UniProt_name==name,['MachineLearning_score']].values.tolist()[0])
        mine.append(pred.loc[pred.name==name,['surface']].values.tolist()[0])
        mineS.append(pred.loc[pred.name==name,['score']].values.tolist()[0])
out = np.column_stack((mismatch,theirs,theirS,mine,mineS))
out = pd.DataFrame(data=out, columns=['name','surfy',
                                      'surfy_score','surfy2','surfy2_score'])
out = out.sort_values(by='surfy2_score',ascending=False)
out.to_csv('mismatches.csv',encoding='utf-8',index=False)

# generate a .csv file with 10 most confident SURFY2 discrepancies for both classes
out_max = out.iloc[:10,:]
out_min = out.iloc[-10:,:]
out2 = pd.concat([out_max,out_min],axis=0)
out2.to_csv('most_confident_mismatches.csv',encoding='utf-8',index=False)
        
