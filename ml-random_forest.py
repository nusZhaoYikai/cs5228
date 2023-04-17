import pandas as pd
from matplotlib import pyplot as plt
from openfe import openfe
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
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
            if column in ('resale_price', 'year', 'month'):
                continue
            numeric_transformer = StandardScaler()  # 11974.23419
            # numeric_transformer = MinMaxScaler()    # 14364.67082
            data[column] = numeric_transformer.fit_transform(data[column].values.reshape(-1, 1))

    # Handling missing values (mean padding is used here)
    data = data.fillna(data.mean())

    return data


# Preprocess the data
train_data_with_aux = pd.read_csv('data/train_with_aux.csv')

# 将month分成year, month   12166.65583,确定了应该减小交叉验证的误差
# train_data_with_aux['year'] = train_data_with_aux['month'].apply(lambda x: int(x.split('-')[0]))
# 去除month
# train_data_with_aux['month'] = train_data_with_aux['month'].apply(lambda x: int(x.split('-')[1]))


# 去除bolck和street_name   11973.72038
train_data_with_aux = train_data_with_aux.drop(columns=['block', 'street_name'])
# 去除eco_category
train_data_with_aux = train_data_with_aux.drop(columns=['eco_category'])  # 11797.76167
# 去除elevation
train_data_with_aux = train_data_with_aux.drop(columns=['elevation'])  # 11797.76167

train_data_processed = preprocess_data(train_data_with_aux)

# 特征筛选，去除相关性较低的特征
delete_columns = [
    # "nearest_commercial_centre_distance",
    # "subzone",
    # "town",
    # "planning_area",
    # "nearest_hawker_centre_distance",
    # "nearest_primary_school_distance",
    "nearest_secondary_school_distance"]

train_data_processed = train_data_processed.drop(
    columns=delete_columns)

# shuffle the data
train_data_processed = train_data_processed.sample(frac=1, random_state=42)

# Split the data set
X = train_data_processed.drop(columns=['resale_price'])

# 打印eco_category的种类
# print(X['month'].unique())
# print(len(X['month'].unique()))
y = train_data_processed['resale_price']
# Scale the target variable
scal = 1e4 / 2
y = y / scal


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
# for n_estimators in [150, 200, 250, 300, 350, 400, 450, 500]:
#     print(f"Training model with Random Forest with {n_estimators} estimators...")
#     model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
#     model.fit(X_train, y_train)
#
#     # evaluate the model
#     y_pred = model.predict(X_val)
#     mse = mean_squared_error(y_val, y_pred)
#     r2 = r2_score(y_val, y_pred)
#     print(f'Mean squared error: {mse:.2f}')
#     print(f'R^2 score: {r2:.2f}')


# print(f"Training model with Random Forest with {n_estimators} estimators...")

# Cross validation

def cross_validation(X, y, n_estimators=1000):
    mse_scores = []
    r2_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 11804.67933

    best_model = None
    best_mse = float('inf')
    best_split_indices = None
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        ofe = openfe()
        # 将X_train和y_train转化为DataFrame
        # X_train = pd.DataFrame(X_train)
        # y_train = pd.DataFrame(y_train)
        ofe.fit(pd.DataFrame(X_train), pd.DataFrame(y_train), n_jobs=60)
        X_train, X_val = ofe.transform(X_train, X_val, ofe.new_features_list[:50], n_jobs=64)

        print(f"Training model with Random Forest with {n_estimators} estimators...")
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=64, bootstrap=True, random_state=42,
                                      min_samples_leaf=1,
                                      # max_depth=100,
                                      max_features='auto')
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mse_scores.append(mse)
        r2_scores.append(r2)
        print(f'Mean squared error: {mse:.2f}')
        print(f'R^2 score: {r2:.2f}')

        # Save the model with the highest R^2 score
        if mse < best_mse:
            best_mse = mse
            best_split_indices = (train_index, val_index)

    print(f'Minimum mean squared error: {best_mse:.2f}')

    # Calculate average MSE and R^2 scores
    avg_mse = sum(mse_scores) / len(mse_scores)
    avg_r2 = sum(r2_scores) / len(r2_scores)
    print(f'Average mean squared error: {avg_mse:.2f}')
    print(f'Average R^2 score: {avg_r2:.2f}')


n_estimators = 2000
# cross_validation(X, y, n_estimators=n_estimators)
# Train the model on the best split
# train_index, val_index = best_split_indices
# X_train, X_val = X.iloc[train_index], X.iloc[val_index]
# y_train, y_val = y.iloc[train_index], y.iloc[val_index]

# ofe = openfe()
# ofe.fit(pd.DataFrame(X), pd.DataFrame(y), n_jobs=60)
# train_size = int(len(X) * 0.8)
# X_train, X_val = X[:train_size], X[train_size:]
# X_train, X_val = ofe.transform(X_train, X_val, ofe.new_features_list[:20], n_jobs=60)
# # 合并X_train和X_val
# X = pd.concat([X_train, X_val], axis=0)
#
# best_model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)  # full data 11801.21165 without any delete
# best_model.fit(X, y)
# 查看特征重要性
#
# feature_importances = pd.DataFrame(best_model.feature_importances_,
#                                    index=X.columns, )
# feature_importances.columns = ['importance']
# feature_importances.sort_values(by='importance', ascending=False, inplace=True)
# print(feature_importances)
# # 可视化特征重要性
# feature_importances.plot(kind='barh', figsize=(10, 10))
# plt.show()
# plt.savefig('feature_importance.png')
# plt.close()
# Train the model on the entire dataset
# best_model.fit(X, y)
#
# Use the model to make predictions

test_data = pd.read_csv('data/test_with_aux.csv')
test_data = test_data.drop(columns=['block', 'street_name'])
test_data = test_data.drop(columns=['eco_category'])
test_data = test_data.drop(columns=['elevation'])
# test_data['year'] = test_data['month'].apply(lambda x: int(x.split('-')[0]))
# test_data['month'] = test_data['month'].apply(lambda x: int(x.split('-')[1]))

test_data_processed = preprocess_data(test_data)

test_data_processed = test_data_processed.drop(
    columns=delete_columns)

# test_data_processed_A, test_data_processed_B = test_data_processed[:len(test_data_processed) // 2], \
#     test_data_processed[len(test_data_processed) // 2:]
# test_data_processed_A, test_data_processed_B = ofe.transform(test_data_processed_A, test_data_processed_B,
#                                                              ofe.new_features_list[:20], n_jobs=60)
ofe = openfe()
ofe.fit(pd.DataFrame(X), pd.DataFrame(y), n_jobs=60)
# train_size = int(len(X) * 0.8)
# X_train, X_val = X[:train_size], X[train_size:]
print(f"feature number: {len(ofe.new_features_list)}")
X, test_data_processed = ofe.transform(X, test_data_processed, ofe.new_features_list[:30], n_jobs=60)
# 合并X_train和X_val


best_model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=64, bootstrap=True, random_state=42,
                                      min_samples_leaf=1,
                                      # max_depth=100,
                                      max_features='auto')  # full data 11801.21165 without any delete
best_model.fit(X, y)

predictions = best_model.predict(test_data_processed)
predictions = predictions * scal

sample_submission = pd.read_csv('data/sample-submission.csv')
sample_submission['Predicted'] = predictions
sample_submission.to_csv('data/submission_random.csv', index=False)
