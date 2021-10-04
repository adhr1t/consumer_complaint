import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('complaints.csv')


df = df[['Product','Consumer complaint narrative']]
df.rename(columns={'Consumer complaint narrative':'Consumer_complaint_narrative'}, inplace = True)

# drop rows with missing values in Consumer complaint narrative column bc we need values to test our model with
df.dropna(subset = ['Consumer_complaint_narrative'], inplace = True)

# reset indices
df.index = range(len(df))    

# Encode Product column
df['Category_ID'] = LabelEncoder().fit_transform(df['Product'])
  

#df_out = df.to_csv('complaints_cleaned.csv', index = False)
