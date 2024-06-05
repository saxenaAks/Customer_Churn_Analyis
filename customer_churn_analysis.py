import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate(username="your_username", key="your_api_key")

# Download the customer churn dataset from Kaggle
api.dataset_download_files('blastchar/telco-customer-churn')
df = pd.read_csv('telco-customer-churn.zip')

# Data Cleaning and Transformation
df = df.dropna()
df['join_date'] = pd.to_datetime(df['join_date'])
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
df['tenure_days'] = (df['last_purchase_date'] - df['join_date']).dt.days
df = pd.get_dummies(df, columns=['contract_type', 'customer_service_interaction'])

# Exploratory Data Analysis (EDA)
plt.hist(df['tenure_days'], bins=30)
plt.xlabel('Tenure Days')
plt.ylabel('Frequency')
plt.title('Distribution of Customer Tenure')
plt.show()
contract_churn = df.groupby('contract_type_Monthly')['churn'].mean()
print(contract_churn)

# Machine Learning Model Training
X = df.drop(['customer_id', 'churn'], axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Model Interpretation and Reporting
importances = rf_classifier.feature_importances_
feature_names = X.columns
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Predicting Customer Churn')
plt.show()
report = f"Model Accuracy: {accuracy}\nFeature Importances:\n{dict(zip(feature_names, importances))}"
print(report)

# Save the model for future use
joblib.dump(rf_classifier, 'churn_prediction_model.pkl')
