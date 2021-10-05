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
  

df.Product.value_counts()


## combine Products that are the same to reduce data complexity

# change "Credit Reporting" products to "Credit reporting, credit repair services, or other personal consumer reports"
df["Product"] = df["Product"].map(lambda x: "Credit reporting, credit repair services, or other personal consumer reports" if str(x).lower().strip() == "Credit reporting".lower().strip() else x)

# change "Credit Card" products to "Credit card or prepaid card"
df["Product"] = df["Product"].map(lambda x: "Credit card or prepaid card" if str(x).lower().strip() == "Credit card".lower().strip() else x)

# change "Prepaid Card" products to "Credit card or prepaid card"
df["Product"] = df["Product"].map(lambda x: "Credit card or prepaid card" if str(x).lower().strip() == "Prepaid card".lower().strip() else x)

# change "Payday Loan" products to "Payday loan, title loan, or personal loan"
df["Product"] = df["Product"].map(lambda x: "Payday loan, title loan, or personal loan" if str(x).lower().strip() == "Payday loan".lower().strip() else x)

# change "Money transfers" products to "Money transfer, virtual currency, or money service"
df["Product"] = df["Product"].map(lambda x: "Money transfer, virtual currency, or money service" if str(x).lower().strip() == "Money transfers".lower().strip() else x)

# change "Virtual currency" products to "Money transfer, virtual currency, or money service"
df["Product"] = df["Product"].map(lambda x: "Money transfer, virtual currency, or money service" if str(x).lower().strip() == "Virtual currency".lower().strip() else x)


# Encode Product column
df['Category_ID'] = LabelEncoder().fit_transform(df['Product'])


# df_out = df.to_csv('complaints_cleaned.csv', index = False)
# df_out = df.to_csv('complaints_cleaned_comb.csv', index = False)
