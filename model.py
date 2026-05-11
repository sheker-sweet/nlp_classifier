# Data handling
import pandas as pd
import numpy as np
import re
import emoji

# Plotting
import matplotlib.pyplot as plt


# Machine Learning
from sklearn.model_selection import  TimeSeriesSplit #we are using timeseriessplit instead of random split to prevent from future data leaking which might result in fake accuracy 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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



# SELECTING FEATURES AND TARGET VARIABLES
# select the features and target variables from the csv files 
X = merged[['avg_sentiment', 'avg_positive', 'avg_negative', 'avg_neutral', 'tweet_count']]
y = merged['direction']


# =========================
# LOGISTIC REGRESSION
# =========================
scaler = StandardScaler()
tscv = TimeSeriesSplit(n_splits=5)



lr_precisions = []
lr_recalls = []
lr_f1s = []
lr_accuracies = []

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

    lr_accuracies.append(accuracy_score(y_val, preds))
    lr_precisions.append(precision_score(y_val, preds))
    lr_recalls.append(recall_score(y_val, preds))
    lr_f1s.append(f1_score(y_val, preds))



print('*' * 100)
print(' '* 100)
print("Logistic Regression Cross Validation Accuracy:", np.mean(lr_accuracies))
print("Logistic Regression Cross Validation Precision:", np.mean(lr_precisions))
print("Logistic Regression Cross Validation Recall:", np.mean(lr_recalls))
print("Logistic Regression Cross Validation F1:", np.mean(lr_f1s))


folds = range(1, len(lr_accuracies) + 1)

plt.figure(figsize=(8,5))

plt.plot(folds, lr_accuracies, marker='o', label='Accuracy')
plt.plot(folds, lr_precisions, marker='o', label='Precision')
plt.plot(folds, lr_recalls, marker='o', label='Recall')
plt.plot(folds, lr_f1s, marker='o', label='F1 Score')

plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Logistic Regression CV Metrics")
plt.ylim(0, 1)

plt.legend()
plt.grid(True)

plt.show()

# Train/test split
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

print('*' * 100)
print(' '* 100)

lr_final_acc = accuracy_score(y_test, preds)
lr_final_prec = precision_score(y_test, preds, average='binary')
lr_final_recall = recall_score(y_test, preds,average='binary' )
lr_final_f1 = f1_score(y_test, preds, average='binary')

print('Logistic Regression Final test accuracy:', lr_final_acc)
print('Logistic Regression Final test precision :', lr_final_prec)
print('Logistic Regression Final test recall', lr_final_recall)
print('Logistic Regression Final test f-1 score', lr_final_f1 )
print('*' * 100 )
print(' '* 100)
print('Logistic Regression Classification Report:', '\n'*2, )

cm_lr = confusion_matrix(y_test, preds)
print(cm_lr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr)

disp.plot(cmap='Blues')

plt.title("Logistic Regression Confusion Matrix")
plt.show()



# timeseries split of the data to preserve the chronology 



# =========================
# DECISION TREE
# =========================

dt_precisions = []
dt_recalls = []
dt_f1s = []
dt_accuracies = []
for train_idx, test_idx in tscv.split(X):

    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    # NO SCALING needed
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)

    preds = dt.predict(X_val)
    dt_accuracies.append(accuracy_score(y_val, preds))
    dt_precisions.append(precision_score(y_val, preds))
    dt_recalls.append(recall_score(y_val, preds))
    dt_f1s.append(f1_score(y_val, preds))



print('*' * 100 )
print(' '* 100)
print("Decision Tree Cross Validation Accuracy:", np.mean(dt_accuracies))
print("Decision Tree Cross Validation Precision:", np.mean(dt_precisions))
print("Decision Tree Cross Validation Recall:", np.mean(dt_recalls))
print("Decision Tree Cross Validation F1:", np.mean(dt_f1s))

folds = range(1, len(dt_accuracies) + 1)

plt.figure(figsize=(8,5))

plt.plot(folds, dt_accuracies, marker='o', label='Accuracy')
plt.plot(folds, dt_precisions, marker='o', label='Precision')
plt.plot(folds, dt_recalls, marker='o', label='Recall')
plt.plot(folds, dt_f1s, marker='o', label='F1 Score')

plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Decision Tree CV Metrics")
plt.ylim(0, 1)

plt.legend()
plt.grid(True)

plt.show()

# # scale (fit ONLY on train)
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# model
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# evaluate on TRUE test set
preds = model.predict(X_test)

print('*' * 100)
print(' '* 100)

dt_final_acc = accuracy_score(y_test, preds)
dt_final_prec = precision_score(y_test, preds, average='binary')
dt_final_recall = recall_score(y_test, preds,average='binary' )
dt_final_f1 = f1_score(y_test, preds, average='binary')

print('Decision Tree Final test accuracy:', dt_final_acc)
print('Decision Tree Final test precision:', dt_final_prec)
print('Decision Tree Final test recall:', dt_final_recall)
print('Decision Tree Final test f1-score:', dt_final_f1)
print('*' * 100 )
print(' '* 100)
print('Decision Tree Classification report:', '\n'*2, classification_report(y_test, preds))

cm_dt = confusion_matrix(y_test, preds)
print(cm_dt)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt)

disp.plot(cmap='Greens')

plt.title("Decision Tree Confusion Matrix")
plt.show()


# =========================
# KNN
# =========================

knn_precisions = []
knn_recalls = []
knn_f1s = []
knn_accuracies = []

for train_idx, test_idx in tscv.split(X):

    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    # SCALE (critical for KNN)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # KNN model
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train_scaled, y_train)

    # Evaluation across folds 
    preds = knn.predict(X_val_scaled)
    knn_accuracies.append(accuracy_score(y_val, preds))
    knn_precisions.append(precision_score(y_val, preds))
    knn_f1s.append(f1_score(y_val, preds))
    knn_recalls.append(recall_score(y_val, preds))


print('*' * 100 )
print(' '* 100)
print("KNN Cross Validation Accuracy:", np.mean(knn_accuracies))
print("KNN Cross Validation Precision:", np.mean(knn_precisions))
print("KNN Cross Validation Recall:", np.mean(knn_recalls))
print("KNN Cross Validation F1:", np.mean(knn_f1s))

folds = range(1, len(knn_accuracies) + 1)

plt.figure(figsize=(8,5))

plt.plot(folds, knn_accuracies, marker='o', label='Accuracy')
plt.plot(folds, knn_precisions, marker='o', label='Precision')
plt.plot(folds, knn_recalls, marker='o', label='Recall')
plt.plot(folds, knn_f1s, marker='o', label='F1 Score')

plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("KNN CV Metrics")
plt.ylim(0, 1)

plt.legend()
plt.grid(True)

plt.show()

# scale (fit ONLY on train)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model
model = KNeighborsClassifier(n_neighbors=15)
model.fit(X_train_scaled, y_train)

# evaluate on TRUE test set
preds = model.predict(X_test_scaled)

print('*'* 100)
print(' '* 100)
knn_final_acc = accuracy_score(y_test, preds)
knn_final_prec = precision_score(y_test, preds, average='binary')
knn_final_recall = recall_score(y_test, preds,average='binary' )
knn_final_f1 = f1_score(y_test, preds, average='binary')

print('KNN Final test accuracy:', knn_final_acc)
print('KNN Final test precision:',knn_final_prec)
print('KNN Final test recall:', knn_final_recall)
print('KNN Final test f1-score:', knn_final_f1)
print('*' * 100 )
print(' '* 100)
print('KNN Final classification report:', '\n'*2, classification_report(y_test, preds))


cm_knn = confusion_matrix(y_test, preds)
print(cm_knn)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn)

disp.plot(cmap='Reds')

plt.title("KNN Confusion Matrix")
plt.show()


# =========================
# NEURAL NETWORK 
# =========================

# build the model 

nn_accuracies = []
nn_precisions = []
nn_recalls = []
nn_f1s = []

for train_idx, test_idx in tscv.split(X):

    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    # SCALE (critical for KNN)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # BUILD FRESH MODEL EACH FOLD
    model1 = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model1.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # train
    model1.fit(
        X_train_scaled,
        y_train,
        epochs=20,
        batch_size=16,
        # verbose=0
    )

    # predict probabilities
    probs = model1.predict(np.array(X_val_scaled), verbose=0)

    # convert probabilities to 0/1
    preds = (probs > 0.5).astype(int).flatten()

    nn_accuracies.append(accuracy_score(y_val, preds))
    nn_precisions.append(precision_score(y_val, preds))
    nn_recalls.append(recall_score(y_val, preds))
    nn_f1s.append(f1_score(y_val, preds))

print('*' * 100 )
print(' '* 100)
print("Feed forward neural network CV Accuracy:", np.mean(nn_accuracies))
print("Feed forward neural network CV Precision:", np.mean(nn_precisions))
print("Feed forward neural network CV Recall:", np.mean(nn_recalls))
print("Feed forward neural network CV F1:", np.mean(nn_f1s))

folds = range(1, len(nn_accuracies) + 1)

plt.figure(figsize=(8,5))

plt.plot(folds, nn_accuracies, marker='o', label='Accuracy')
plt.plot(folds, nn_precisions, marker='o', label='Precision')
plt.plot(folds, nn_recalls, marker='o', label='Recall')
plt.plot(folds, nn_f1s, marker='o', label='F1 Score')

plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("NN CV Metrics")
plt.ylim(0, 1)

plt.legend()
plt.grid(True)

plt.show()

# build a second model 

model2 = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

model2.compile(
        optimizer=Adam(learning_rate=0.001), #i could increase the learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
model2.fit(
        X_train_scaled,
        y_train,
        epochs=20,
        batch_size=16,
        # verbose=0
    )

probs = model2.predict(np.array(X_test_scaled), verbose=0)

preds = (probs > 0.5).astype(int)

print('*' * 100 )
print(' '* 100)
nn_final_acc = accuracy_score(y_test, preds)
nn_final_prec = precision_score(y_test, preds, average='binary')
nn_final_recall = recall_score(y_test, preds,average='binary' )
nn_final_f1 = f1_score(y_test, preds, average='binary')

print('Feed forward neural network Final test accuracy:', nn_final_acc)
print('Feed forward neural network Final test precision:', nn_final_prec)
print('Feed forward neural network Final test recal:', nn_final_recall)
print('Feed forward neural network Final test f-1 score', nn_final_f1)
print('*' * 100 )
print(' '* 100)
print('Feed forward neural network Classification Report:', '\n'*2, classification_report(y_test, preds))

cm_nn = confusion_matrix(y_test, preds)
print(cm_nn)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_nn)

disp.plot(cmap='Purples')

plt.title("NN Confusion Matrix")
plt.show()



# =========================
# EVALUATION ACCROSS MODELS 
# =========================

# CROSS VALIDATION VALUES 
cv_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Neural Network'],
    
    'CV Accuracy': [
        np.mean(lr_accuracies),
        np.mean(knn_accuracies),
        np.mean(dt_accuracies),
        np.mean(nn_accuracies)
    ],

    'CV Precision': [
        np.mean(lr_precisions),
        np.mean(knn_precisions),
        np.mean(dt_precisions),
        np.mean(nn_precisions)
    ],

    'CV Recall': [
        np.mean(lr_recalls),
        np.mean(knn_recalls),
        np.mean(dt_recalls),
        np.mean(nn_recalls)
    ],

    'CV F1': [
        np.mean(lr_f1s),
        np.mean(knn_f1s),
        np.mean(dt_f1s),
        np.mean(nn_f1s)
    ]
})

print('*' * 100 )
print(' '* 100)
print(cv_results)

# FINAL VALUES 

final_results = pd.DataFrame({
    'Model': ['LR', 'KNN', 'DT', 'NN'],
    'Accuracy': [lr_final_acc, knn_final_acc, dt_final_acc, nn_final_acc],
    'Precision': [lr_final_prec, knn_final_prec, dt_final_prec, nn_final_prec],
    'Recall': [lr_final_recall, knn_final_recall, dt_final_recall, nn_final_recall],
    'F1': [lr_final_f1, knn_final_f1, dt_final_f1, nn_final_f1]
})

print('*' * 100 )
print(' '* 100)
print(final_results.round(3))

# PLOT CROSS VALIDATION RESULTS
cv_results.plot(
    x='Model',
    y=['CV Accuracy', 'CV Precision', 'CV Recall', 'CV F1'],
    kind='bar',
    figsize=(10,6)
)

plt.title('Cross-Validation Metrics by Model')
plt.ylabel('Score')
plt.xlabel('Model')

plt.ylim(0,1)

plt.xticks(rotation=0)

plt.legend(loc='lower right')

plt.tight_layout()

plt.show()

# PLOT FINAL VALUES 

final_results.plot(
    x='Model',
    y=['Accuracy', 'Precision', 'Recall', 'F1'],
    kind='bar',
    figsize=(10,6)
)

plt.title('Final Test Metrics by Model')
plt.ylabel('Score')
plt.xlabel('Model')

plt.ylim(0,1)

plt.xticks(rotation=0)

plt.legend(loc='lower right')

plt.tight_layout()

plt.show()








