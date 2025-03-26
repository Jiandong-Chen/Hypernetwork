import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.model import *

batch_size = 256

data = pd.read_csv("./data/adult.csv")

# data_cleaning
non_numeric_columns = data.select_dtypes(include=['object']).columns
numeric_columns = data.select_dtypes(exclude=['object']).columns

#Limit categorization
data.loc[:, 'marital-status'] = data['marital-status'].replace({
    'Never-married': 'NotMarried',
    'Married-AF-spouse': 'Married',
    'Married-civ-spouse': 'Married',
    'Married-spouse-absent': 'NotMarried',
    'Separated': 'Separated',
    'Divorced': 'Separated',
    'Widowed': 'Widowed'
})

data.loc[:, 'education'] = data['education'].replace({
    'Preschool': 'dropout',
    '10th': 'dropout',
    '11th': 'dropout',
    '12th': 'dropout',
    '1st-4th': 'dropout',
    '5th-6th': 'dropout',
    '7th-8th': 'dropout',
    '9th': 'dropout',
    'HS-Grad': 'HighGrad',
    'HS-grad': 'HighGrad',
    'Some-college': 'CommunityCollege',
    'Assoc-acdm': 'CommunityCollege',
    'Assoc-voc': 'CommunityCollege',
    'Bachelors': 'Bachelors',
    'Masters': 'Masters',
    'Prof-school': 'Masters',
    'Doctorate': 'Doctorate'
})


#remove duplicated row
data=data.drop_duplicates()

#replace ? to nan
data.replace('?', np.nan, inplace=True)

df = data

#drop beacuse they have nan
df['occupation'].dropna(inplace=True)
df['workclass'].dropna(inplace=True)

#drop educational-num beacuse its not important
df = df.drop(['educational-num'],axis=1)

#Encoder cetegorical columns
lb=LabelEncoder()
df.workclass=lb.fit_transform(df.workclass)
df.education=lb.fit_transform(df.education)
df['marital-status']=lb.fit_transform(df['marital-status'])
df.occupation=lb.fit_transform(df.occupation)
df.relationship=lb.fit_transform(df.relationship)
df.race=lb.fit_transform(df.race)
df.gender=lb.fit_transform(df.gender)
df['native-country']=lb.fit_transform(df['native-country'])
df.income=lb.fit_transform(df.income)

X = df.drop('income',axis=1)
y = df['income']

st=StandardScaler()
X_scaled=st.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long).squeeze()

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 500
best_val_loss = float('inf')
best_model_state = None

model = MLP(input_dim = X.shape[1], num_layers=10, output_dim = 2)
# model = MainMLP(input_dim, hidden_dim, output_dim)
# model = MainMLP_random_z(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects logits and integer labels
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


# Track metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
learning_rates = []

# Training loop
num_epochs = 100
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for batch in train_loader:
        X_batch, y_batch = batch

        optimizer.zero_grad()
        outputs = model(X_batch)  # Logits
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        predictions = torch.argmax(outputs, dim=1)  # Get class with highest probability
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss /= total
    train_accuracy = correct / total

    # Validation loop
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = batch
            outputs = model(X_batch)  # Logits
            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    val_loss /= total
    val_accuracy = correct / total

    # If validation accuracy increased, save the model
    # if val_accuracy > best_val_accuracy:
    #     print(f"Validation accuracy increased from {best_val_accuracy:.4f} to {val_accuracy:.4f}. Saving current model.")
    #     best_val_accuracy = val_accuracy
    #     best_model_state = model.state_dict()  # Save current model

    if val_loss < best_val_loss:
        print(f"Validation loss decrease from {best_val_loss:.4f} to {val_loss:.4f}. Saving current model.")
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # Save current model

    else:
        train_loss = train_losses[-1] if train_losses else train_loss
        val_loss = val_losses[-1] if val_losses else val_loss
        train_accuracy = train_accuracies[-1] if train_accuracies else train_accuracy
        val_accuracy = val_accuracies[-1] if val_accuracies else val_accuracy

    # Step the scheduler
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

    # Save metrics for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} - "
          f"LR: {current_lr:.6f}")

# After training, load the best model
model.load_state_dict(best_model_state)
print("Best model loaded based on validation accuracy.")

# Plotting train and validation loss and accuracy curves
epochs = np.arange(1, num_epochs + 1)

# Plot train and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Val Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

# Set custom y-axis ticks for loss plot (step size 0.01)
y_min = min(min(train_losses), min(val_losses))  # Get the minimum loss value
y_max = max(max(train_losses), max(val_losses))  # Get the maximum loss value
plt.yticks(np.arange(y_min, y_max, 0.01))  # Set y-ticks from min to max with a step of 0.01

# Plot train and validation accuracy
plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
plt.plot(epochs, val_accuracies, label='Val Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, learning_rates, marker='o', linestyle='-', color='green')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.savefig('learning_rate_schedule.png')

# Save the plots
plt.tight_layout()
plt.savefig('train_val_loss_accuracy_original.png')