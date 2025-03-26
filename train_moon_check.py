import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from models.model import *


def generate_nonlinear_highdim_data(n_samples=5000, n_features=20, noise=0.1):
    np.random.seed(42)
    
    # Base XOR pattern in 2D
    X_base = np.random.randn(n_samples, 2)
    y = (X_base[:, 0] * X_base[:, 1] > 0).astype(int)  # XOR-like pattern

    # Introduce nonlinear transformations and extra features
    extra_features = np.random.randn(n_samples, n_features - 2)  # Random extra features
    X_nonlinear = np.c_[
        X_base,  
        np.sin(2 * np.pi * X_base[:, [0]]),  # Sin transform
        np.cos(2 * np.pi * X_base[:, [1]]),  # Cos transform
        X_base[:, [0]] ** 2,  # Quadratic term
        X_base[:, [1]] ** 3,  # Cubic term
        extra_features
    ]
    
    # Add noise to labels
    flip_mask = np.random.rand(n_samples) < noise
    y[flip_mask] = 1 - y[flip_mask]  # Flip some labels to add difficulty
    
    return X_nonlinear, y

# Create dataset
X, y = generate_nonlinear_highdim_data(n_samples=20000, n_features=20, noise=0.1)

st=StandardScaler()
X_scaled=st.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long).squeeze()

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# model = MLP(input_dim = X.shape[1], num_layers=3, output_dim = 2)
# model = MainMLP(input_dim = X.shape[1], hidden_dim = 100, output_dim = 2)
# model = MainMLP_random_z(input_dim = X.shape[1], hidden_dim = 100, output_dim = 2)
model = MainMLP_random_z_check(input_dim = X.shape[1], hidden_dim = 100, output_dim = 2)
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
        w_init = torch.randn_like(X_batch)
        outputs = model(X_batch, w_init)  # Logits
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
            w_init_test = torch.randn_like(X_batch)
            outputs = model(X_batch, w_init_test)  # Logits
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

val_loss, correct, total = 0.0, 0, 0
model.eval()
with torch.no_grad():
    for batch in val_loader:
        X_batch, y_batch = batch
        torch.manual_seed(42)
        w_init_test = torch.randn_like(X_batch)
        outputs = model(X_batch, w_init_test)  # Logits
        loss = criterion(outputs, y_batch)

        val_loss += loss.item() * X_batch.size(0)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    val_loss /= total
    val_accuracy = correct / total

print(f"Final Evaluation"
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} - "
        f"LR: {current_lr:.6f}")

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
plt.savefig('train_val_loss_accuracy_hyper_check.png')