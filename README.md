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
*   Reduced the dataset to only Consumer Complaint Narratives and Product columns
*   dropped all rows in which Consumer Complaint Narratives were left blank or unavailable
*   Coalesced similar Product classifications such as "Payday loan" &rarr; "Payday loan, title loan, or personal loan" by reassigning Product classifications as the holistic, overarching Product classification
*   Label Encoded Product values

# EDA
Built a histogram to visualize data spread and class imbalance. Also outputted the most common unigrams and bigrams per Product classification.

![histogram](https://user-images.githubusercontent.com/72672768/136120792-0b315453-3107-4c02-bd29-08eb9b57bbb9.png)

An example of the most frequent unigrams and bigrams for certain Product classifications:

![unigrams](https://user-images.githubusercontent.com/72672768/136121009-addcc510-fe77-420d-99e7-d46a03425c24.png)

# Model Building
I split the data into train and test sets with a test size of 20%. I then trained a Multinomial Naive Bayes model as a benchmark, and then trained Random Forest Classifier, Linear SVM, and Logistic Regression classifier models.\ 
I evaluated the models based on their prediction Accuracy.   
*   **Multinomial Naive Bayes:** ..5834
*   **Random Forest Classifier:** .4190
*   **Linear SVM:** .7761
*   **Logistic Regression:** .7411
