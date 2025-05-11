Python 3.13.2 (v3.13.2:4f8bb3947cf, Feb  4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
... import pandas as pd
... import numpy as np
... from sklearn.model_selection import train_test_split
... from sklearn.preprocessing import StandardScaler, LabelEncoder
... from sklearn.cluster import KMeans
... from sklearn.decomposition import PCA
... import seaborn as sns
... import matplotlib.pyplot as plt
... import pickle
... 
... # Load the dataset
... df = pd.read_csv('Dataset (ATS).csv')
... 
... # Display basic info about the dataset
... print("Initial Data Overview:")
... print(df.info())
... print(df.describe())
... 
... # Data Cleaning: Handling Missing Values
... print("\nMissing Values:")
... print(df.isnull().sum())
... 
... # Impute missing values (example with forward fill)
... df.fillna(method='ffill', inplace=True)
... 
... # Verify that missing values are handled
... print("\nMissing Values After Imputation:")
... print(df.isnull().sum())
... 
... # Encode Categorical Variables
... categorical_cols = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'Churn']
... for col in categorical_cols:
...     le = LabelEncoder()
...     df[col] = le.fit_transform(df[col])
... 
print("\nData After Encoding Categorical Variables:")
print(df.head())

# Split the dataset into training and testing sets
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display 10 rows of the scaled training data
df_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
print("\n10 Rows of Prepared and Normalized Training Data:")
print(df_scaled.head(10))

# Plotting distributions before and after scaling for a feature (e.g., 'MonthlyCharges')
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['MonthlyCharges'], kde=True, color='skyblue')
plt.title('MonthlyCharges Distribution Before Scaling')
plt.xlabel('MonthlyCharges')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(X_train_scaled[:, X.columns.get_loc('MonthlyCharges')], kde=True, color='salmon')
plt.title('MonthlyCharges Distribution After Scaling')
plt.xlabel('MonthlyCharges (scaled)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Clustering Analysis

# Finding the optimal number of clusters using the elbow method
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    sse.append(kmeans.inertia_)

# Plotting the elbow method
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.grid(True)
plt.show()

# Using KMeans for clustering with optimal k (let's assume k=4 based on the elbow method)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)

# Save the trained model to a file
with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

# Adding cluster labels to the dataset
df['Cluster'] = kmeans.predict(scaler.transform(X))

# Visualizing the resulting clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(14, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title(f'Customer Segments with {optimal_k} Clusters (Using PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar()
plt.show()

print("Data preparation and clustering analysis complete. Check the visualizations for results.")


# Import necessary libraries for building the ANN model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score

# Load and preprocess the dataset
df = pd.read_csv('Dataset (ATS).csv')

# Encode categorical variables
categorical_cols = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'Churn']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Split the dataset into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the architecture of the ANN model
ann_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(units=64, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=16, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = ann_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Predict churn on the test data
y_pred = (ann_model.predict(X_test_scaled) > 0.5).astype(int)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report: \n{classification_report_output}")


