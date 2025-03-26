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
y = df['pctid']

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).squeeze()

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

model = MLP(input_dim = X.shape[1], num_layers=3, hidden_dim=100, output_dim = 1)
# model = MainMLP(input_dim = X.shape[1], hidden_dim = 100, output_dim = 2)
# model = MainMLP_random_z(input_dim = X.shape[1], hidden_dim = 100, output_dim = 1)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


# Track metrics
train_losses, val_losses = [], []
train_maes, val_maes = [], []
train_r2s, val_r2s = [], []
learning_rates = []

# Training loop
num_epochs = 100
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    train_loss, train_preds, train_targets = 0.0, [], []

    for batch in train_loader:
        X_batch, y_batch = batch

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()  # Logits
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        train_preds.extend(outputs.detach().cpu().numpy())
        train_targets.extend(y_batch.cpu().numpy())

    # Compute training metrics
    train_loss /= len(train_loader.dataset)
    train_mae = mean_absolute_error(train_targets, train_preds)
    train_r2 = r2_score(train_targets, train_preds)

    # Validation loop
    model.eval()
    val_loss, val_preds, val_targets = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = batch
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)

    if val_loss < best_val_loss:
        print(f"Validation loss decrease from {best_val_loss:.4f} to {val_loss:.4f}. Saving current model.")
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # Save current model

    else:
        train_loss = train_losses[-1] if train_losses else train_loss
        val_loss = val_losses[-1] if val_losses else val_loss
        train_mae = train_maes[-1] if train_maes else train_mae
        val_mae = val_maes[-1] if val_maes else val_mae
        train_r2 = train_r2s[-1] if train_r2s else train_r2
        val_r2 = val_r2s[-1] if val_r2s else val_r2

    # Step the scheduler
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

    # Save metrics for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_maes.append(train_mae)
    train_r2s.append(train_r2)
    val_maes.append(val_mae)
    val_r2s.append(val_r2)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f} - "
          f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f} - "
          f"LR: {current_lr:.6f}")

# After training, load the best model
model.load_state_dict(best_model_state)
print("Best model loaded based on validation accuracy.")

# Plotting train and validation loss and accuracy curves
epochs = np.arange(1, num_epochs + 1)

# Plot train and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Val Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

# Plot train and validation accuracy
plt.subplot(1, 4, 2)
plt.plot(epochs, train_maes, label='Train MAE', color='blue')
plt.plot(epochs, val_maes, label='Val MAE', color='red')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Train and Validation MAE')
plt.legend()

# Plot train and validation accuracy
plt.subplot(1, 4, 3)
plt.plot(epochs, train_r2s, label='Train R-Square', color='blue')
plt.plot(epochs, val_r2s, label='Val R-Square', color='red')
plt.xlabel('Epochs')
plt.ylabel('R-Square')
plt.title('Train and Validation R-Square')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(epochs, learning_rates, marker='o', linestyle='-', color='green')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

# Save the plots
plt.tight_layout()
plt.savefig('./figures/Tabular/accele_reg_base.png')


