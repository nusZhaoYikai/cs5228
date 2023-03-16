import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from dl_models import BiLSTM

# Set a random seed
torch.manual_seed(42)
np.random.seed(42)

torch.cuda.manual_seed(42)

# Load data
df = pd.read_csv('data/train.csv')

# Feature engineering
df['age'] = 2023 - df['lease_commence_date']
df['floor_height'] = df['storey_range'].str.extract('(\d+)').astype(int)

# Iterates through each column of df and converts it to category type if the data type of the column is not numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')

# Shuffle the data
df = df.sample(frac=1, random_state=42)

# Divide the dataset
X = df.drop('resale_price', axis=1)
scal = 1e2 / 2
y = df['resale_price'] / scal

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Data preprocessing Defines
# converters for numeric and categorical variables
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Define a Column Transformer object to apply different converters to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer,
         ['floor_area_sqm', 'age', 'floor_height', 'latitude', 'longitude', 'elevation', 'lease_commence_date']),
        ('cat', categorical_transformer,
         ['month', 'town', 'flat_type', 'block', 'street_name', 'storey_range', 'flat_model', 'eco_category', 'subzone',
          'planning_area', 'region'])
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Convert to the PyTorch tensor
X_train = torch.Tensor(X_train.toarray())
y_train = torch.Tensor(y_train.values).view(-1, 1)
X_test = torch.Tensor(X_test.toarray())
y_test = torch.Tensor(y_test.values).view(-1, 1)


# Train the model
def train(model, optimizer, criterion, dataloader, scale_factor=1 / 1024):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(dataloader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # The loss function is scaled
        scaled_loss = loss * scale_factor
        if scaled_loss > 10000:
            print('Scale factor:', scale_factor, 'scaled loss:', scaled_loss.item())
            scale_factor /= 2  # Shrink the gradient

        total_loss += scaled_loss.item()

        # Backpropagation
        optimizer.zero_grad()
        scaled_loss.backward()
        optimizer.step()

        # Restore the gradient size
        for param in model.parameters():
            param.grad /= scale_factor

    return total_loss / len(dataloader), scale_factor


def evaluate(model, criterion, dataloader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()  # sum up batch loss
    return loss / len(dataloader)


# See how much data there is in total
print('Train data size:', len(X_train))

batch_size = 128
epochs = 200
learning_rate = 0.001
hidden_dim = 1024
dropout_prob = 0.3



train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Use the Bi LSTM model
model = BiLSTM(input_dim=X_train.shape[1], hidden_dim=hidden_dim, output_dim=1, dropout_prob=dropout_prob)

model.to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []
scale_factor = 1 / 1024
best_loss = 1e10

for epoch in range(1, epochs + 1):
    train_loss, scale_factor = train(model, optimizer, criterion, train_dataloader, scale_factor=scale_factor)

    test_loss = evaluate(model, criterion, test_dataloader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'Epoch {epoch}, scaled_train_loss: {train_loss:.5f}, test_loss: {test_loss:.5f}')
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'model.pth')

plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Sets the scale of the x-axis
plt.xticks(np.arange(1, epochs + 1, 1))
# # Set the abscissa from 1
plt.xlim(1, epochs)

plt.title('Loss vs Epoch')
plt.savefig('loss.png')
plt.show()

# Predict house prices
# Load the model
model.load_state_dict(torch.load('model.pth'))

df_test = pd.read_csv('data/test.csv')
df_test['age'] = 2023 - df_test['lease_commence_date']
df_test['floor_height'] = df_test['storey_range'].str.extract('(\d+)').astype(int)

X_test = preprocessor.transform(df_test)
X_test = torch.Tensor(X_test.toarray())
test_dataset = TensorDataset(X_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()

predictions = []

with torch.no_grad():
    for data in test_dataloader:
        data = data[0].to(device)
        output = model(data)
        predictions.extend(output.cpu().numpy().flatten())

predictions = np.array(predictions) * scal


sample_submission = pd.read_csv('data/sample-submission.csv')

sample_submission['Predicted'] = predictions


sample_submission.to_csv('data/submission-lstm.csv', index=False)

