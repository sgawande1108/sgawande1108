# NLP- Disaster Tweets
### 1. Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm  import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import warnings
warnings.filterwarnings('ignore')
### 2. Load Dataset
train_data=pd.read_csv('train.csv')
train_data.head()
train_data.shape
test_data=pd.read_csv('test.csv')
test_data.head()
test_data.shape
train_data.info()
test_data.info()
train_data.duplicated().sum()
- Showing unique values and number of unique values in each column
train_data.nunique()
-  Show Distribution of 2 Values in target Column

plt.figure(figsize=(10,5))
ax=sns.countplot(x='target',data=train_data,palette='pastel')
plt.title('Distribution of Target Column')

total = len(train_data['target'])
for p in ax.patches:
    height = p.get_height()  # Get the height of each bar
    percentage = f'{100 * height / total:.2f}%'  # Calculate percentage
    ax.annotate(percentage,                      # The text to display
                (p.get_x() + p.get_width() / 2., height),  # Position the text
                ha='center', va='bottom')        # Center it

plt.show()
-  Checking and Handling missing Values

train_data.isna().sum()

- Observation:

- There are null values in 2 columns keyword,location.
# Show Percentage of null values in keyword,location

train_data['location'].isna().sum()/train_data.shape[0]*100
train_data['keyword'].isna().sum()/train_data.shape[0]*100
# Visualization of Null values

plt.figure(figsize=(10,5))
sns.heatmap(data=train_data.isna(),yticklabels=False,cbar=False,cmap='viridis')
plt.xticks(rotation=40)
plt.show()
- Observation:
- As id Column isn't usefull at all,I will drop it.

- Location Column has 33.27 % Null value,So I decided to drop it.

- keyword Column which has 0.8 % Null value and It includs only one word from tweet,So I see it's useless and I decided to remove it.

- The main aim is to classify tweet if's disaster or not,So I will keep only 2 columns which represents this problem: text,target.
train_data.drop(columns='id',inplace=True)
train_data.drop(columns=['location','keyword'],inplace=True)
train_data
# Check for Null Values and Duplicated Rows After Removing id,keyword,location Columns

train_data.isna().sum()

train_data.duplicated().sum()
- We observe that:
1. There isn't any null value in 2 columns.

2. There are 92 duplicated rows,So we must drop them all.
train_data.drop_duplicates(inplace=True)
# Checking that duplicates are removed

train_data.duplicated().sum()
train_data.shape
### Preprocessing Tweets Text
train_data.info()
train_data['text']
stop_words=set(stopwords.words('english'))
#lemmatization=WordNetLemmatizer()
stemming=PorterStemmer()
def clean_text(text):
    # 1. Convert to lower
    text = text.lower()

    # 2. Remove special characters (e.g., #, @, emojis)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keeps only letters and spaces

    # 3. Split to words
    tokens = word_tokenize(text)

    # 4. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Remove numbers
    tokens = [word for word in tokens if not word.isdigit()]

    # 6. Apply Lemmatization
    #tokens = [lematization.lemmatize(word) for word in tokens]

    # 7. Apply Stemming
    tokens = [stemming.stem(word) for word in tokens]

    # To return these single words back into one string
    return ' '.join(tokens)
train_data['cleaned_text']=train_data['text'].apply(clean_text)
# Show tweets after cleaning
train_data
#### Converting Text into Numbers before Modeling
tf_idf=TfidfVectorizer()
# Identify input and output before splitting
x=tf_idf.fit_transform(train_data['cleaned_text'])
y=train_data['target']
# Splitting Data
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
### Modeling Using ML Algorithms
#### 1. Logistic Regression
log_r=LogisticRegression()
log_r.fit(X_train,y_train)
logr_y_pred=log_r.predict(X_test)
accuracy = accuracy_score(y_test, logr_y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report of Logistic Regression:\n", classification_report(y_test, logr_y_pred))
print('Training accuracy:', log_r.score(X_train,y_train))
print('Test accuracy:', log_r.score(X_test,y_test))
sns.heatmap(confusion_matrix(y_test,logr_y_pred), annot=True, fmt="d")
#### 2. Random Forest
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf_y_pred=rf.predict(X_test)
accuracy = accuracy_score(y_test, rf_y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report of Random Forest:\n", classification_report(y_test, rf_y_pred))
print('Training accuracy:', rf.score(X_train,y_train))
print('Test accuracy:', rf.score(X_test,y_test))
sns.heatmap(confusion_matrix(y_test,rf_y_pred), annot=True, fmt="d")
#### 3. Naive Bayes
nb=MultinomialNB()
nb.fit(X_train,y_train)
nb_y_pred=nb.predict(X_test)
accuracy = accuracy_score(y_test, nb_y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report of Naive Bayes:\n", classification_report(y_test, logr_y_pred))
print('Training accuracy:', nb.score(X_train,y_train))
print('Test accuracy:', nb.score(X_test,y_test))
sns.heatmap(confusion_matrix(y_test,nb_y_pred), annot=True, fmt="d")
#### 4.SVC
svc=SVC()
svc.fit(X_train,y_train)
svc_y_pred=svc.predict(X_test)
accuracy = accuracy_score(y_test, svc_y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report of SVC:\n", classification_report(y_test, svc_y_pred))
print('Training accuracy:', svc.score(X_train,y_train))
print('Test accuracy:', svc.score(X_test,y_test))
sns.heatmap(confusion_matrix(y_test,svc_y_pred), annot=True, fmt="d")
- Observation on Modeling:¶
1. we notice that there is an obvious Overfitting using all 4 ML Models,As all mosels have almost trainning accuracy between 97, 99 % and testing accuracy 79 %.

2. So I will apply Grid Search and Cross Validation to prevent Overfitting.
### Applying Grid Search to Reduce or Prevent Overfitting


#### 1. Grid Search on Logistic Regression
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.001,0.01,0.1,1,10,100]}
grid_search_lr=GridSearchCV(log_r,param_grid,cv=5,scoring='accuracy')
grid_search_lr.fit(X_train,y_train)
print(f"Best C: {grid_search_lr.best_params_['C']}")
# Take the best model of logistic with the best parameter to calculate training,testing accuracy
best_model_lr=grid_search_lr.best_estimator_

# Training Accuracy
y_train_pred_lr=best_model_lr.predict(X_train)
trainning_accuracy_lr=accuracy_score(y_train,y_train_pred_lr)

# Testing Accuracy
y_test_pred_lr=best_model_lr.predict(X_test)
testing_accuracy_lr=accuracy_score(y_test,y_test_pred_lr)

print(f"Training Accuracy of Logistic Regression : {trainning_accuracy_lr * 100:.2f}")
print(f"Testing Accuracy  of Logistic Regression: {testing_accuracy_lr * 100:.2f}")
print("Classification Report  of Logistic Regression:\n", classification_report(y_test, y_test_pred_lr))
#### 2.Grid Search on Random Forest
param_grid={'n_estimators':[50,100,200],
            'max_depth':[5,10,20,None],
            'min_samples_split':[2,5,10,20]
           }
grid_search_rf=GridSearchCV(rf,param_grid,cv=5,scoring='accuracy')
grid_search_rf.fit(X_train,y_train)
print(" Best params of Random Forest: ",grid_search_rf.best_params_)
# Take the best model of Random Forest with the best parameter to calculate training,testing accuracy
best_model_rf=grid_search_rf.best_estimator_

# Training Accuracy
y_train_pred_rf=best_model_rf.predict(X_train)
trainning_accuracy_rf=accuracy_score(y_train,y_train_pred_rf)

# Testing Accuracy
y_test_pred_rf=best_model_rf.predict(X_test)
testing_accuracy_rf=accuracy_score(y_test,y_test_pred_rf)

print(f"Training Accuracy of Random Forest : {trainning_accuracy_rf * 100:.2f}")
print(f"Testing Accuracy  of Random Forest : {testing_accuracy_rf * 100:.2f}")
print("Classification Report of Random Forest after Grid Search:\n",
      classification_report(y_test, y_test_pred_rf))
- Observation on Random Forest After Grid Search:

  1.Testing Accuracy improved 1% after applying Grid Search on Ravdom Forest.

 2. OverFitting still occurs.
#### 3.Grid Search on SVC
param_grid={
    'C':[0.001,0.01,0.1,1,10,],
    'class_weight': [None, 'balanced']
           }
grid_search_svc=GridSearchCV(svc,param_grid,cv=5,scoring='accuracy')
grid_search_svc.fit(X_train,y_train)
print(" Best params of SVC: ",grid_search_svc.best_params_)
# Take the best model of logistic with the best parameter to calculate training,testing accuracy
best_model_svc=grid_search_svc.best_estimator_

# Training Accuracy
y_train_pred_svc=best_model_svc.predict(X_train)
trainning_accuracy_svc=accuracy_score(y_train,y_train_pred_svc)

# Testing Accuracy
y_test_pred_svc=best_model_svc.predict(X_test)
testing_accuracy_svc=accuracy_score(y_test,y_test_pred_svc)

print(f"Training Accuracy of SVC : {trainning_accuracy_svc * 100:.2f}")
print(f"Testing Accuracy  of  SVC: {testing_accuracy_svc * 100:.2f}")
print("Classification Report of SVC after Grid Search:\n",
      classification_report(y_test, y_test_pred_svc))
- Observation on SVC after Grid Search:

  1.Testing Accuracy improved 1% after applying Grid Search on Ravdom Forest.

2. OverFitting still occurs.
### Results
#### 1. Logistic Regression

- Training accuracy: 0.8896276595744681
- Test accuracy: 0.7880398671096346
    
#### Grid Search on Logistic Regression

- Training Accuracy of Logistic Regression : 88.96
- Testing Accuracy  of Logistic Regression: 78.80
#### 2. Random Forest

- Training accuracy: 0.9971742021276596
- Test accuracy: 0.772093023255814
    
#### Grid Search on Random Forest

- Training Accuracy of Random Forest : 97.71
- Testing Accuracy  of Random Forest : 78.14
#### 3. Naive Bayes

- Training accuracy: 0.9005984042553191
- Test accuracy: 0.7873754152823921
    
#### 4. SVC

- Training accuracy: 0.9720744680851063
- Test accuracy: 0.7833887043189369
    
#### Grid Search on SVC

- Training Accuracy of SVC : 98.02
- Testing Accuracy  of  SVC: 79.14

