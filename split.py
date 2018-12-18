# Goal: Prepare Train, Validation and Test datasets
# How: Split them into Train/Test and then Train/Validation
import pandas as pd
import random
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
random.seed(42)
df = pd.read_csv('train_data.csv')

# split into train / test data
train,test = train_test_split(df, shuffle=True, random_state=42,
                              test_size=0.1)

# split train data into train / validation data
train,val = train_test_split(train,shuffle=True,random_state=42,
                             test_size=0.1)
train.to_csv('train.csv',encoding='utf-8',index=False)
val.to_csv('val.csv',encoding='utf-8',index=False)
test.to_csv('test.csv',encoding='utf-8',index=False)
