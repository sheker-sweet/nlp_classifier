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
from sklearn.pipeline import make_pipeline
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # add more evaluation tools for the nlp component


# Sentiment analysis
import nltk

from nltk.corpus import words
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize



nltk.download('words', quiet = True) #not sure if the better way to do is nltk.download('all)
nltk.download('punkt', quiet = True)
nltk.download('vader_lexicon', quiet = True )


#import the data 

tweets = pd.read_csv('stock_tweets.csv')
spy = pd.read_csv('spy_prices.csv')
# spy['Date'] = pd.to_datetime(spy['Date']).dt.date # not sure what this line of code is doing
prices = spy.copy()

# pre-process the data 
def clean_tweet(tweet):
    text = emoji.demojize(tweet) # convert emojis to text
    text = text.lower() # convert to lowercase
    text = re.sub(r'http\S+', '', text) # remove urls
    text = re.sub(r'@\w+', '', text) # remove mentions
    text = re.sub(r'#\w+', '', text) # remove hashtags
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# steps for tweets: 
# 1) handle missing data; 
tweets.dropna(subset=['Tweet'], inplace=True)
# 2) pares dates; 
tweets['Date'] = pd.to_datetime(tweets['Date']).dt.date
# 3)clean the tweet text; 
tweets['clean_tweet'] = tweets['Tweet'].apply(clean_tweet)
# 5) apply vader sentiment;
analyzer = SentimentIntensityAnalyzer()

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

# 6) aggregate by day 
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
#_____________________________________________________________________________________________________________________________

# steps for prices: 
# 1) handle missing dates 
# 2) parse and sort dates; 
# 3) clean the target variable
# merge two datasets

# import yfinance as yf
# import time 

# time.sleep(120)






# steps for prices: 
# 1) parse + sort dates
# your file uses Date_ / Close_SPY, so normalize to Date / Close first
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



merged = pd.merge(daily_sentiment, prices[['Date', 'direction']], on='Date', how='inner')
merged.sort_values('Date', inplace=True)
merged.reset_index(drop=True, inplace=True)

print(f"Merged rows:  {len(merged)}")

# merge two datasets
# merged = pd.merge(daily_sentiment, prices[['Date', 'direction']], on='Date')
# merged.sort_values('Date', inplace=True)  # keep chronological order




# select the features and target variables from the csv files 

X = merged[['avg_sentiment', 'avg_positive', 'avg_negative', 'avg_neutral', 'tweet_count']]
y = merged['direction']


# # checking for the balance of the classes in target variable
# print(y.value_counts())
# print(y.value_counts(normalize=True))

# # Checking if Vader scores vary day to day 
# print(merged['avg_sentiment'].describe())
# print(merged['avg_sentiment'].plot())

# # Checking if there's a corellation between sentiment and price movement (direction)
# print(merged[['avg_sentiment', 'direction']].corr())

# Check your column names first
# print(prices.columns.tolist())

# Then filter for SPY or S&P 500 only
# depending on what your ticker column is called

# If column is called 'Stock Name'
# prices_sp500 = prices[prices['Stock Name'] == 'SPY']

# Or if called 'Ticker'
# prices_sp500 = prices[prices['Ticker'] == 'SPY']

# Check result
# print(f"Filtered prices rows: {len(prices_sp500)}")
# print(f"Unique dates:         {prices_sp500['Date'].nunique()}")
# print(prices_sp500.head())



# print(prices['Stock Name'].unique())
# print(prices['Stock Name'].value_counts())




# timeseries split of the data to preserve the chronology 

tscv = TimeSeriesSplit(n_splits=5) # not sure if it needs to be split into two

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # fix: .iloc
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # fix: .iloc

# Chronological 80/20 split for final train/test evaluation
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# for i, (train_index, test_index) in enumerate(tscv.split(X)):
#     print(f"Fold {i}:")
#     print(f"  Train: index={train_index}")
#     print(f"  Test:  index={test_index}")
# logisitc regression model 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=0, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# lr_pipeline = make_pipeline(
#     StandardScaler(),
#     LogisticRegression(random_state=0, max_iter=1000)
# )
scores = cross_val_score(lr_model, X, y, cv=tscv, scoring='accuracy') #you can use the pipeline instead of the model if you want to scale the data before fitting the model

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print(scores.std()) #low value --> model is consistent; high-value--> model is inconsistent

# extract predicted and actual labels

y_pred = lr_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred) #--> maybe one for each model
print("Confusion matrix (test set):")
print(cm)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm_norm).plot(values_format='.2f')
plt.show()

# decision tree classifier
# dt_model = DecisionTreeClassifier()
# dt_scores = cross_val_score(dt_model, X, y, cv=tscv, scoring='accuracy')


# # knn classifier 
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_acc = knn_model.score(X_test_scaled, y_test)
print(f"KNN test accuracy: {knn_acc:.2f}")
# knn_cm = confusion_matrix(y_test, knn_pred)
print("KNN confusion matrix (test set):")
# print(knn_cm)

# # Create model
# knn_model = KNeighborsClassifier(n_neighbors=5)

# # Train model
# knn_model.fit(X_train_scaled, y_train)

# # Predictions
# knn_pred = knn_model.predict(X_test_scaled)

# # Accuracy
# knn_acc = knn_model.score(y_test, knn_pred)

# print(f"KNN test accuracy: {knn_acc:.2f}")

# # Confusion matrix
# knn_cm = confusion_matrix(y_test, knn_pred)

# print("KNN confusion matrix (test set):")
# print(knn_cm)

# add 

# feed forward neural network 

# model = Sequential()
# model.add(Dense(20, input_dim=X_shape[1], activation = 'relu')) #input dimension will be equal to the number of columns in the X
# model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='sgd',
#               loss='mse',
#               metrics=['accuracy']) #optimizer, loss and metrics can be decided 


# evaluation: 5-fold cross validation, f-1 score, accuracy, precision, recall, confusion matrix





# instead of 5 fold-cross validation, use timeseries cross validation b/c accounts for data chronology and prevents future data from leaking into the existing dataset 




