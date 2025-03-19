# GROWTHLINK
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data_url = "https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies"
df = pd.read_csv("movies.csv")  # Replace with actual file path

# Data Exploration
print(df.head())
print(df.info())
print(df.isnull().sum())  # Check for missing values

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.dropna(inplace=True)  # Drop remaining null values if necessary

# Encode categorical variables
label_enc = LabelEncoder()
df['Genre'] = label_enc.fit_transform(df['Genre'])
df['Director'] = label_enc.fit_transform(df['Director'])
df['Actors'] = label_enc.fit_transform(df['Actors'])

# Define features and target
X = df.drop(columns=['IMDB_Rating'])
y = df['IMDB_Rating']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
