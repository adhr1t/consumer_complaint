# Consumer Complaint Classifier: Project Overview
*   Created a tool that predicts the Classifications of Consumer Complaints on Financial products (Accuracy ~ .7761) to help categorize and pinpoint issues
*   Cleaned over 2,200,000 financial complaint documentations from across the United States
*   Reduced data complexity by combining similar Product classifications via methods such as merging the "Credit Reporting" product complaints into the overarching "Credit reporting, credit repair services, or other personal consumer reports" classification
*   Employed TfidfVectorizer and the chi-squared test to determine the most common unigrams and bigrams for each complaint classification
*   Trained Multinomial Naive Bayes, Random Forest Classifier, Linear SVM, and Logistic Regression classifier models
*   Determined Linear SVM was the most efficacious model and evaluated it via Confusion Matrix, Classification Report, and Cross Validation Accuracy score

# Code and Resources Used
**Python Version:** 3.9\
**Packages:**   pandas, numpy, matplotlib, sklearn, and seaborn\
**Data Source:**  https://catalog.data.gov/dataset/consumer-complaint-database

# Cleaning

