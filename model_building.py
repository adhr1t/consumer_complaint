import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB



df = pd.read_csv('complaints_cleaned.csv')
df = df.sample(n=4673)


# create dataframe with all the Product categories
category_id_df = df[['Product', 'Category_ID']].drop_duplicates().sort_values('Category_ID')

# dictionary of Product categories
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['Category_ID', 'Product']].values)


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