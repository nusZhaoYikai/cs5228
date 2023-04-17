import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import numpy as np


# Prepare the data
def preprocess_data(data):
    # Converts categorical features to numeric features
    data['flat_type'] = data['flat_type'].str.replace('-', ' ')
    data['gap'] = data['month'].astype(str).str[:4].astype(np.int64) - data['lease_commence_date']
    data['gap_mean'] = data['gap'].mean()
    data['floor_area_sqm_mean'] = data['floor_area_sqm'].mean()

    for i,column in enumerate(data.columns):
        #print(i, column, data[column].dtype)
        if data[column].dtype == 'object' or column == 'lease_commence_date':
            data[column] = data[column].astype('str')
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
train_data_with_aux = pd.read_csv('data/train_with_aux.csv').drop(columns=['eco_category', 'block', 'elevation', 'street_name'])
train_data_processed = preprocess_data(train_data_with_aux)
# shuffle the data
train_data_processed = train_data_processed.sample(frac=1, random_state=42)

# Split the data set
X = train_data_processed.drop(columns=['resale_price'])
y = train_data_processed['resale_price']
#y = np.log1p(y)
# Scale the target variable
scal = 1e0
y = y / scal

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=5)
test_predictions = []
X = X.values
#print(X[0])
y = y.values
print(X.shape, y.shape)
mean_mae = 0
for i, (train_index, val_index) in enumerate(kf.split(y)):
    #print(train_index)
    X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]

    # train the model
    parameters = {
                  'random_state':42,
                  'boosting': ['dart'],
                  'n_estimators': 40000,
                  'max_depth': 10,
                  'metric': 'mae',
                  #'max_depth': [8, 10, 12],
                  #'num_leaves': list(range(16, 4000, 48)),
                  #'learning_rate': [0.01, 0.02],
                  #'feature_fraction': [0.8, 0.9, 1.0],
                  # 'feature_fraction': 0.8,
                  #'bagging_fraction': [0.8, 0.9. 1.0],
                  #'bagging_freq': [4],
                  #'lambda_l1': 3,
                  #"bagging_freq": 5, "bagging_fraction": 0.75,
                  'lambda_l2': 4,
                  'learning_rate': 0.05,
                  'drop_rate':1/40000,
    }

    print("Training model with LightGBM...")
    print(parameters)
    eval_set = [(X_val, y_val)]
    model = lgb.LGBMRegressor(device='gpu', gpu_platform_id = 0, gpu_device_id = 0, **parameters)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=1000, eval_metric="mae", verbose=True,
              categorical_feature=[1,2,5,11])

    # evaluate the model
    y_pred = model.predict(X_train)
    #y_pred = np.expm1(y_pred)
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    print(f'fold: {i} train Mean squared error: {mse:.6f}')
    print(f'fold: {i} train Mean absolute error: {mae:.6f}')
    print(f'fold: {i} train R^2 score: {r2:.6f}')

    y_pred = model.predict(X_val)
    #y_pred = np.expm1(y_pred)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f'fold: {i} val Mean squared error: {mse:.6f}')
    print(f'fold: {i} val Mean absolute error: {mae:.6f}')
    print(f'fold: {i} val R^2 score: {r2:.6f}')
    mean_mae += mae

    # Use the model to make predictions
    test_data = pd.read_csv('data/test_with_aux.csv').drop(columns=['eco_category', 'block', 'elevation', 'street_name'])
    test_data_processed = preprocess_data(test_data)
    predictions = model.predict(test_data_processed)
    #test_data_processed = np.expm1(test_data_processed)
    predictions = predictions * scal

    test_predictions.append(predictions)

    # Save
    sample_submission = pd.read_csv('data/sample-submission.csv')
    sample_submission['Predicted'] = predictions
    sample_submission.to_csv(f'data/submission_lgbm_{mae:.6f}_{str(parameters)}_fold{i}_0.csv', index=False)


test_predictions = np.array(test_predictions)
print(test_predictions.shape)
test_predictions = np.mean(test_predictions, axis=1)
print(test_predictions.shape)

sample_submission = pd.read_csv('data/sample-submission.csv')

sample_submission['Predicted'] = predictions

mean_mae /= 5
sample_submission.to_csv(f'data/submission_lgbm_{mean_mae:.6f}_{str(parameters)}_0.csv', index=False)

print(parameters)