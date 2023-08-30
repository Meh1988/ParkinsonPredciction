import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM

# Load data
df = pd.read_csv('parkinsons.data')

# Drop the 'name' column
df.drop('name', axis=1, inplace=True)

# Data Preparation
X = df.drop('status', axis=1)
y = df['status']

# Feature Importance using SelectKBest
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

# Plotting Feature Importance
plt.figure(figsize=(12, 6))
plt.barh(X.columns, selector.scores_)
plt.xlabel('Feature Importance Scores')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# To collect the accuracy scores for all models
accuracy_scores = {}

# Traditional Machine Learning Models
models = {
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy}')
    accuracy_scores[name] = accuracy

# Deep Learning Models

# Simple Neural Network
model_nn = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nn.fit(X_train, y_train, epochs=50, batch_size=10)
accuracy_nn = model_nn.evaluate(X_test, y_test)[1]
print(f'Accuracy of Simple Neural Network: {accuracy_nn}')
accuracy_scores['Simple Neural Network'] = accuracy_nn

# Convolutional Neural Network (CNN)
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

model_cnn = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train_cnn, y_train, epochs=50, batch_size=10)
accuracy_cnn = model_cnn.evaluate(X_test_cnn, y_test)[1]
print(f'Accuracy of CNN: {accuracy_cnn}')
accuracy_scores['CNN'] = accuracy_cnn

# Recurrent Neural Network (RNN)
X_train_rnn = np.expand_dims(X_train, axis=2)
X_test_rnn = np.expand_dims(X_test, axis=2)

model_rnn = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1, activation='sigmoid')
])

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_rnn.fit(X_train_rnn, y_train, epochs=50, batch_size=10)
accuracy_rnn = model_rnn.evaluate(X_test_rnn, y_test)[1]
print(f'Accuracy of RNN: {accuracy_rnn}')
accuracy_scores['RNN'] = accuracy_rnn

# Plotting Model Comparison
plt.figure(figsize=(12, 6))
plt.barh(list(accuracy_scores.keys()), list(accuracy_scores.values()))
plt.xlabel('Accuracy Scores')
plt.ylabel('Models')
plt.title('Model Comparison')
plt.show()
