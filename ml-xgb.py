import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Prepare the data
def preprocess_data(data):
    # Converts categorical features to numeric features
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
        # Standard scaling for numeric features, except for resale price
        elif data[column].dtype == 'int64' or data[column].dtype == 'float64':
            if column == 'resale_price':
                continue
            numeric_transformer = StandardScaler()
            data[column] = numeric_transformer.fit_transform(data[column].values.reshape(-1, 1))

    # Handling missing values (mean padding is used here)
    data = data.fillna(data.mean())

    return data


# Preprocess the data
train_data_with_aux = pd.read_csv('data/train_with_aux.csv')
train_data_processed = preprocess_data(train_data_with_aux)
# shuffle the data
train_data_processed = train_data_processed.sample(frac=1, random_state=42)

# Split the data set
X = train_data_processed.drop(columns=['resale_price'])
y = train_data_processed['resale_price']
# Scale the target variable
scal = 1e4
y = y / scal
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training model with XGBoost...")
model = xgb.XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.01, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f'Mean squared error: {mse:.2f}')
print(f'R^2 score: {r2:.2f}')

# Use the model to make predictions
test_data = pd.read_csv('data/test_with_aux.csv')
test_data_processed = preprocess_data(test_data)
predictions = model.predict(test_data_processed)
predictions = predictions * scal

sample_submission = pd.read_csv('data/sample-submission.csv')

sample_submission['Predicted'] = predictions

sample_submission.to_csv('data/submission_xgb.csv', index=False)
