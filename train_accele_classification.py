import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.model import *

df = pd.read_csv("./data/Tabular/accelerometer.csv")

X = df.drop(['wconfid', 'pctid'], axis=1)
y = df['wconfid']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long).squeeze()


X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

model = MLP(input_dim = X.shape[1], num_layers=3, output_dim = 3)
# model = MainMLP(input_dim = X.shape[1], hidden_dim = 100, output_dim = 3)
# model = MainMLP_random_z(input_dim = X.shape[1], hidden_dim = 100, output_dim = 3)
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects logits and integer labels
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


# Track metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
learning_rates = []

# Training loop
num_epochs = 200
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

# Save the plots
plt.tight_layout()
plt.savefig('./figures/Tabular/accele_base.png')