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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = MLP(input_dim = X.shape[1], num_layers=3, output_dim = 2)
# model = MainMLP(input_dim = X.shape[1], hidden_dim = 100, output_dim = 2)
# model = MainMLP_random_z_version1(input_dim = X.shape[1], seed_length = 100, hidden_dim = 100, output_dim = 2)
# model = MainMLP_random_z_version2(input_dim = X.shape[1], seed_length = 100, hidden_dim = 100, output_dim = 2)
model = MainMLP_random_z_version3(input_dim = X.shape[1], seed_length = 100, hidden_dim = 100, output_dim = 2)
model.to(device)
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects logits and integer labels
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


# Track metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
learning_rates = []

# Training loop
num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

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
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
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
        torch.save(model, "best_model.pth")  # Save current model

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
model = torch.load("best_model.pth", weights_only=False)
# model.to(device)
print("Best model loaded based on validation accuracy.")

val_loss, correct, total = 0.0, 0, 0
model.eval()
with torch.no_grad():
    for batch in val_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        val_loss += loss.item() * X_batch.size(0)  # Only if criterion uses mean reduction
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
plt.savefig('train_val_loss_accuracy_3MLP_Gen_feature.png')




