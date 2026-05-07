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
from sklearn.metrics import confusion_matrix # add more evaluation tools for the nlp component


# Sentiment analysis
import nltk
from nltk.corpus import words
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize



nltk.download('words', quiet = True) #not sure if the better way to do is nltk.download('all)
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
tweets['vader_score'] = tweets['clean_tweet'].apply(
    lambda t: analyzer.polarity_scores(t)['compound']
)
# 6) aggregate by day 

# daily_sentiment = tweets.groupby('Date')['vader_score'].mean().reset_index() #will decide later what to do with this
# daily_sentiment.columns = ['Date', 'avg_sentiment']

daily_sentiment = tweets.groupby('Date').agg(
   avg_sentiment = ('compound', 'mean'),
   avg_positive  = ('positive', 'mean'),
   avg_negative  = ('negative', 'mean'),
   avg_neutral   = ('neutral',  'mean'),
   tweet_count   = ('compound', 'count')
).reset_index()


# TEST FOR PRE-PROCESSED TWEETS
print("TWEETS:")
print(daily_sentiment.head(10))
print(f"Sentiment range: {daily_sentiment['avg_sentiment'].min():.2f} to {daily_sentiment['avg_sentiment'].max():.2f}")




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


# TEST FOR PRE-PROCESSED PRICES

print("\nPRICES:")
print(prices[['Date', 'Close', 'direction']].head(10))
print(f"Direction counts (0=down, 1=up):\n{prices['direction'].value_counts()}")




# MERGING DATASETS:



# merge two datasets
# merged = pd.merge(daily_sentiment, prices[['Date', 'direction']], on='Date')
# merged.sort_values('Date', inplace=True)  # keep chronological order



merged = pd.merge(daily_sentiment, prices[['Date', 'direction']], on='Date', how='inner')
merged.sort_values('Date', inplace=True)
merged.reset_index(drop=True, inplace=True)

print(f"Merged rows:  {len(merged)}")



# SELECTING FEATURES AND TARGET VARIABLES

# select the features and target variables from the csv files 

X = merged[['avg_sentiment', 'avg_positive', 'avg_negative', 'avg_neutral', 'tweet_count']]
y = merged['direction']


# =========================
# LOGISTIC REGRESSION
# =========================




# timeseries split of the data to preserve the chronology 

# tscv = TimeSeriesSplit(n_splits=5)

# logisitc regression model 

# model = LogisticRegression(random_state=0).fit(X, y)

# scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')

# print(scores.std()) #low value --> model is consistent; high-value--> model is inconsistent



# cm = confusion_matrix(y, model.predict(X)) --> maybe one for each model 

# decision tree classifier
# dt_model = DecisionTreeClassifier()
# dt_scores = cross_val_score(dt_model, X, y, cv=tscv, scoring='accuracy')


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




