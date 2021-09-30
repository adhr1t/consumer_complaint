import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier



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
tfidfP = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidfP.fit_transform(df.Consumer_complaint_narrative).toarray()
labels = df.Category_ID
features.shape

# iterate through the different products and print their most correlated unigrams and bigrams
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])    # indices of sorted features_chi2 array
  feature_names = np.array(tfidfP.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Product))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-2:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-2:])))
  

# create train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'],  test_size=0.2, random_state=42)
countVect = CountVectorizer(stop_words='english')
X_train_counts = countVect.fit_transform(X_train)
tfidfTrans = TfidfTransformer()
X_train_tfidf = tfidfTrans.fit_transform(X_train_counts)
#tfidf = TfidfVectorizer(stop_words='english')
#X_train_tfidf = tfidf.fit_transform(X_train)



## Model Building

# Multinomial Naive Bayes as a benchmark
MNB = MultinomialNB().fit(X_train_tfidf, Y_train)

# some quick tests to see if our model is working
print(MNB.predict(countVect.transform(["CASHCALL IS ATTEMPTING TO COLLECT ON THE NOTE LOAN THAT DOES NOT BELONG TO ME. IN MY NUMEROUST ATTEMPTS TO REQUEST DEBT\
                                       VALIDATION OF THIS LOAN FROM 2013, RECEIVED WAS PAYMENT LEDGER THAT SHOWS DATE LOAN ISSUED AND SOME CREDIT BUT NO PROOF OF\
                                       PAYMENT, NO ID, NO SS # ON THE CONTRACT, AND CONTRACT IS NOT SIGNED, DOESN'T HAVE MY SS # OR DOB OR ADDRESS. 1-CONTRACT IS\
                                       MISSING SIGNATURE 2- CONTRACT IS MISSING ORIGINATION METHOD AND WHO IS THE PERSON ORIGINATING THIS LOAN 3- CALIFORNIA STATUTORY\
                                       LIMIT FOR INTEREST RATE IS 10 %. THIS IS A LOAN ORIGINATED IN CA, AT 183.06 % ISN'T THIS USURY?. CASHCALL RECEIVED A RESPONCE\
                                       FROM ME DISPUTING THIS DEBT AND INSTEAD OF INVESTIGATING THIS MATTER THEY SOLD THIS ERRONEOUS FRAUDULENT COLLECTION TO CCI\
                                       ACQUISITIONS , LLC WHO IS NOW THREATENING TO SUE ME AND GARNISH MY WAGES ... IST N'T THERE A STATUTORY TIME ALLOWED FOR THEM\
                                       TO TAKE LEGAL ACTION THAT IS IF THIS ACCOUNT WAS VALID AND ACCURATE? .... THIS IS FRAUD ON INSTITUITIONAL LEVEL!"])))
                                       # should predict "Debt Collection"
                                       
print(MNB.predict(countVect.transform(["I contacted Equifax Dispute Center, and was connected with XXXX, one of their customer service reps. I explained to him that\
                                       I have tried many times to remove old addresses from my credit report, and have had no luck. He said that as long as creditors\
                                       report them to Equifax, there is nothing he could do to help me. I stated to him that by law, Equifax is supposed to report correct\
                                       information, and none of these old addresses are correct. He refused to delete them. I next moved on to some old accounts they\
                                       have listed on my report. I informed him that I have written the XXXX XXXX XXXX XXXX twice to try and verify their claims that\
                                       I had late payments on my student loans in 2014 but they do not respond back. I've called them numerous times but nobody can\
                                       verify these late payments, so I asked XXXX to delete them from my report. He said without notice from the creditor, he couldn't\
                                       do that. Same situation with a SSA account that is on my Equifax report. I have contacted them numerous times and they can't\
                                       verify the account belongs to me, which it doesn't, it belongs to my deceased father XXXX XXXX. I asked him to delete this account\
                                       due to lack of verification, but he refused."])))        # should predict "Credit reporting..."
                                       


## testing other models

models = [RandomForestClassifier(n_estimators=200, max_depth=3, random_state=43), LinearSVC(), MultinomialNB(), LogisticRegression(random_state=42)]
cv = 3

                                       