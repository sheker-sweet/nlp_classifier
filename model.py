# Data handling
import pandas as pd
import numpy as np
import re
import emoji

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score #we are using timeseriessplit instead of random split to prevent from future data leaking which might result in fake accuracy 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Sentiment analysis
import nltk
from nltk.corpus import words
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

nltk.download('words', quiet = True) 
nltk.download('punkt', quiet = True)
nltk.download('vader_lexicon', quiet = True)

analyzer = SentimentIntensityAnalyzer()

#import the data 
tweets = pd.read_csv('stock_tweets.csv')
spy = pd.read_csv('spy_prices.csv')
prices = spy.copy() # not sure if I need to make a copy here or not

# ===============================
# DATA PRE-PROCESSING FOR TWEETS:
# ==============================
def clean_tweet(tweet):
    text = emoji.demojize(tweet)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# steps for tweets: 
# 1) handle missing data; 
tweets.dropna(subset=['Tweet'], inplace=True)
# 2) parse dates; 
tweets['Date'] = pd.to_datetime(tweets['Date']).dt.date
# 3)clean the tweet text; 
tweets['clean_tweet'] = tweets['Tweet'].apply(clean_tweet)
# 5) apply vader sentiment;
tweets['compound'] = tweets['clean_tweet'].apply(
   lambda t: analyzer.polarity_scores(t)['compound']
)   
tweets['positive'] = tweets['clean_tweet'].apply(
   lambda t: analyzer.polarity_scores(t)['pos']
)
tweets['negative'] = tweets['clean_tweet'].apply(
   lambda t: analyzer.polarity_scores(t)['neg']
)
tweets['neutral'] = tweets['clean_tweet'].apply(
   lambda t: analyzer.polarity_scores(t)['neu']
)

# Aggregate by day 
daily_sentiment = tweets.groupby('Date').agg(
   avg_sentiment = ('compound', 'mean'),
   avg_positive  = ('positive', 'mean'),
   avg_negative  = ('negative', 'mean'),
   avg_neutral   = ('neutral',  'mean'),
   tweet_count   = ('compound', 'count')
).reset_index()

# TEST FOR PRE-PROCESSED TWEETS
# print("TWEETS:")
# print(daily_sentiment.head(10))
# print(f"Sentiment range: {daily_sentiment['avg_sentiment'].min():.2f} to {daily_sentiment['avg_sentiment'].max():.2f}")

# ================================
# PRE-PROCESSING STEPS FOR PRICES:
# ================================

# steps for prices: 
# 1) parse + sort dates
# prices['Date'] = pd.to_datetime(prices['Date']).dt.date
# prices.sort_values('Date', inplace=True)
# # 3) create direction label (1 = price up, 0 = price down)
# prices['direction'] = (prices['Close'] > prices['Close'].shift(1)).astype(int)
# prices.dropna(inplace=True)

if 'Date_' in prices.columns:
   prices = prices.rename(columns={'Date_': 'Date'})
if 'Close_SPY' in prices.columns:
   prices = prices.rename(columns={'Close_SPY': 'Close'})

prices.dropna(subset=['Date', 'Close'], inplace=True)
prices['Date'] = pd.to_datetime(prices['Date']).dt.date
prices.sort_values('Date', inplace=True)
# 3) create direction label (1 = price up, 0 = price down)
prices['direction'] = (prices['Close'].shift(-1) > prices['Close']).astype(int)
prices = prices.iloc[:-1].copy()  # drop last row (no next-day label)

#===============================================================
# TEST FOR PRE-PROCESSED PRICES
# print("\nPRICES:")
# print(prices[['Date', 'Close', 'direction']].head(10))
# print(f"Direction counts (0=down, 1=up):\n{prices['direction'].value_counts()}")

# MERGING DATASETS:
merged = pd.merge(daily_sentiment, prices[['Date', 'direction']], on='Date', how='inner')
merged.sort_values('Date', inplace=True) # keep chronological order
merged.reset_index(drop=True, inplace=True)

#print(f"Merged rows:  {len(merged)}")

# SELECTING FEATURES AND TARGET VARIABLES
# select the features and target variables from the csv files 
X = merged[['avg_sentiment', 'avg_positive', 'avg_negative', 'avg_neutral', 'tweet_count']]
y = merged['direction']


# =========================
# LOGISTIC REGRESSION
# =========================
scaler = StandardScaler()
tscv = TimeSeriesSplit(n_splits=5)

# for train_idx, test_idx in tscv.split(X):
#     X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

scores = []
for train_idx, test_idx in tscv.split(X):

    # split data
    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    # scale (fit ONLY on train fold)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # model
    model = LogisticRegression(random_state=0, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # evaluate
    preds = model.predict(X_val_scaled)
    score = accuracy_score(y_val, preds)

    scores.append(score)

print("CV mean accuracy:", np.mean(scores))
print('CV standard deviation', np.std(scores))

# split
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

# scale (fit ONLY on train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model
model = LogisticRegression(random_state=0, max_iter=1000)
model.fit(X_train_scaled, y_train)

# evaluate on TRUE test set
preds = model.predict(X_test_scaled)
score = accuracy_score(y_test, preds)

print("Final test accuracy:", score)
print(accuracy_score(y_test, preds))
print(precision_score(y_test, preds, average='binary'))
print(recall_score(y_test, preds, average='binary'))
print(f1_score(y_test, preds, average='binary'))

print(classification_report(y_test, preds))

# print("CV Accuracy:", np.mean(accuracies))
# print("CV Precision:", np.mean(precisions))
# print("CV Recall:", np.mean(recalls))
# print("CV F1:", np.mean(f1s))


# timeseries split of the data to preserve the chronology 


# logisitc regression model 

# model = LogisticRegression(random_state=0).fit(X, y)

# scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')

# print(scores.std()) #low value --> model is consistent; high-value--> model is inconsistent


# cm = confusion_matrix(y, model.predict(X)) --> maybe one for each model 
# =========================
# DECISION TREE
# =========================
dt_scores = []
for train_idx, test_idx in tscv.split(X):

    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    # NO SCALING needed
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)

    preds = dt.predict(X_val)
    dt_scores.append(accuracy_score(y_val, preds))

print("Decision Tree CV accuracy:", np.mean(dt_scores))

# scale (fit ONLY on train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train_scaled, y_train)

# evaluate on TRUE test set
preds = model.predict(X_test_scaled)
score = accuracy_score(y_test, preds)

print("Final test accuracy:", score)

print(accuracy_score(y_test, preds))
print(precision_score(y_test, preds, average='binary'))
print(recall_score(y_test, preds, average='binary'))
print(f1_score(y_test, preds, average='binary'))

print(classification_report(y_test, preds))

# decision tree classifier
# dt_model = DecisionTreeClassifier()
# dt_scores = cross_val_score(dt_model, X, y, cv=tscv, scoring='accuracy')


# =========================
# KNN
# =========================

knn_scores = []

for train_idx, test_idx in tscv.split(X):

    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    # SCALE (critical for KNN)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    preds = knn.predict(X_val_scaled)
    knn_scores.append(accuracy_score(y_val, preds))

print("KNN CV accuracy:", np.mean(scores))


# scale (fit ONLY on train)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# evaluate on TRUE test set
preds = model.predict(X_test_scaled)
score = accuracy_score(y_test, preds)

print("Final test accuracy:", score)

print(accuracy_score(y_test, preds))
print(precision_score(y_test, preds, average='binary'))
print(recall_score(y_test, preds, average='binary'))
print(f1_score(y_test, preds, average='binary'))

print(classification_report(y_test, preds))

# knn classifier 
# knn_model = KNeighborsClassifier(n_neighbors=1) #need to determine the number of neighbors 
# knn_scores = cross_val_score(knn_model, X, y, cv=tscv, scoring='accuracy')

# cm_knn = confusion_matrix(y, knn_model.predict(X))

# feed forward neural network 

# model = Sequential()
# model.add(Dense(20, input_dim=X_shape[1], activation = 'relu')) #input dimension will be equal to the number of columns in the X
# model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='sgd',
#               loss='mse',
#               metrics=['accuracy']) #optimizer, loss and metrics can be decided 


# evaluation: 5-fold cross validation, f-1 score, accuracy, precision, recall, confusion matrix





# instead of 5 fold-cross validation, use timeseries cross validation b/c accounts for data chronology and prevents future data from leaking into the existing dataset 




