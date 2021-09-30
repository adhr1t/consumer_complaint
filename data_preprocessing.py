import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2


df = pd.read_csv('complaints.csv')


df = df[['Product','Consumer complaint narrative']]
df.rename(columns={'Consumer complaint narrative':'Consumer_complaint_narrative'}, inplace = True)

# drop rows with missing values in Consumer complaint narrative column bc we need values to test our model with
df.dropna(subset = ['Consumer_complaint_narrative'], inplace = True)

#take a sample of the dataset to make it manageable given hardware restraints
df = df.sample(n=14658)

# reset indices
df.index = range(len(df))    

# Encode Product column
df['Category_ID'] = LabelEncoder().fit_transform(df['Product'])

# create dataframe with all the Product categories
category_id_df = df[['Product', 'Category_ID']].drop_duplicates().sort_values('Category_ID')

# dictionary of Product categories
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['Category_ID', 'Product']].values)


## EDA
# plot frequencies
fig = plt.figure(figsize=(8,6))
df.groupby('Product').Consumer_complaint_narrative.count().plot.bar(ylim=0)
plt.show()

df.Product.value_counts()   
# we're seeing lots of bias towards credit reporting, debt collection, and mortgage
# we can fix this with resampling via SMOTE etc. but we want the algorithm to have high accuracy in predicting the 
# majority class, so we'll leave the training data as is


## Text Representation
# Term Frequency, Inverse Document Frequency; tf-idf
# calculated tf-idf vector for each of the Consumer Complaint texts
# min_df=5 so word must be in 5 documents to be kept, ngram_range=(1,2) so unigrams and bigrams are considered, stop_words=english so pronouns like
# 'a' and 'the' are removed/not considered 
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Consumer_complaint_narrative).toarray()
labels = df.Category_ID
features.shape

# iterate through the different products and print their most correlated unigrams and bigrams
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])    # indices of sorted features_chi2 array
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Product))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-2:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-2:])))
  

#df_out = df.to_csv('complaints_cleaned.csv', index = False)
