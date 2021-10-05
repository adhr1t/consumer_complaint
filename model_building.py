import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


#df = pd.read_csv('complaints_cleaned.csv')
df = pd.read_csv('complaints_cleaned_comb.csv')
df = df.sample(n=4673)

# create dataframe with all the Product categories
category_id_df = df[['Product', 'Category_ID']].drop_duplicates().sort_values('Category_ID')

# dictionary of Product categories
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['Category_ID', 'Product']].values)


## Text Representation
# Term Frequency, Inverse Document Frequency; tf-idf
# calculates tf-idf vector for each of the Consumer Complaint texts
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



## Model Building

# Multinomial Naive Bayes as a benchmark
MNB = MultinomialNB().fit(X_train_tfidf, Y_train)

# some quick tests to see if our model is working
print(MNB.predict(countVect.transform(["They never informed me I had a right to dispute. I requested verification of debt and never received it. They continue to call me and\
                                       say they will continue to attempt to recover debt that can not be verified."])))
                                       # should predict "Debt Collection
                                       
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

models = [RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42), LinearSVC(), MultinomialNB(), LogisticRegression(random_state=42)]
cv = 3
cv_df = pd.DataFrame(index=range(cv * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=cv)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
                                       
# visualize model performances
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

# print model average accuracies; SVC performed the best
cv_df.groupby('model_name').accuracy.mean()


## SVC Model Evaluation

modelSVC = LinearSVC(random_state = 42)
X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=42)
modelSVC.fit(X_train, Y_train)
Y_pred_SVC = modelSVC.predict(X_test)

# Confusion Matrix
confMatrix = confusion_matrix(Y_test, Y_pred_SVC)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confMatrix, annot=True, fmt='d', xticklabels=category_id_df.Product.values, yticklabels=category_id_df.Product.values, cmap = 'coolwarm')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# Classification Report; Accuracy included in this and confirmed by cross validation
prods = pd.Series(df['Product'].sort_values().unique())
prods = prods.drop([8]) # drop the 'Other financial service' value (index 8) in the series. Doing this bc model doesn't predict
# anything to be 'Other financial service' so there are no scores to be calculated in the calssification report and it throws an error

try:
    print(classification_report(Y_test, Y_pred_SVC, target_names=df['Product'].unique()))
except:
    print(classification_report(Y_test, Y_pred_SVC, target_names = prods))
else:
    print(classification_report(Y_test, Y_pred_SVC))


# Accuracy 
svc_cv_acc_score = np.mean(cross_val_score(modelSVC, X_train, Y_train, cv=3, scoring= 'accuracy'))    # Accuracy score is .7761



# print mispredictions and their correct Product classification
from IPython.display import display
for predicted in category_id_df.Category_ID:
  for actual in category_id_df.Category_ID:
    if predicted != actual and confMatrix[actual, predicted] >= 10:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], confMatrix[actual, predicted]))
      display(df.loc[indices_test[(Y_test == actual) & (Y_pred_SVC == predicted)]][['Product', 'Consumer_complaint_narrative']])
      print('')


# print most common terms for Products determined from SVC
modelSVC.fit(features, labels)
N = 2
for Product, category_id in sorted(category_to_id.items()):
  indices = np.argsort(modelSVC.coef_[category_id])
  feature_names = np.array(tfidfP.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(Product))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))