import numpy as np
import pandas as pd


train_data = pd.read_table('train.txt', sep='\s+', header=None) #sep='   '
print(train_data)
a = train_data.iloc[1]
print(a)
#print(file)