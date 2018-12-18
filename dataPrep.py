# Goal: Build a training dataset
# How: Select negative & positive training samples used in the PNAS paper from the whole dataset
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
random.seed(42)
# getting whole data & neg/pos train subsets
df = pd.read_excel('data.xlsx')
neg=pd.read_csv('neg.csv')
neg2 = neg.values.tolist()
neg2 = [val for sublist in neg2 for val in sublist]
pos=pd.read_csv('pos.csv')
pos2 = pos.values.tolist()
pos2 = [val for sublist in pos2 for val in sublist]
rows = neg.shape[0]+pos.shape[0]
col = df.shape[1]

# get features for the proteins used for training
negD = df[df['name'].isin(neg2)]
posD = df[df['name'].isin(pos2)]

# create 'surface' label as classification output
ypre = [0]*657+[1]*910
ypre = np.asarray(ypre)
y = pd.DataFrame(data=ypre,columns=['surface'])
out = pd.concat([negD,posD],axis=0)
out['surface']=y.values
out.to_csv('train_data.csv',encoding='utf-8',index=False)
