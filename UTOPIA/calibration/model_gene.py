import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AsymmetricMSELoss(nn.Module):
    def __init__(self, beta=2.0):
        """
        beta: Weight for underestimation penalty (beta > 1 penalizes underestimation more)
        """
        super().__init__()
        self.beta = beta
        
    def forward(self, pred, target):
        diff = target - pred
        mask = diff > 0  # Find where model underestimates
        loss = torch.mean(mask * (self.beta * diff ** 2) + (~mask) * (diff ** 2))
        return torch.sqrt(loss)  # Return RMSE
        
class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta
        
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim=200, output_dim=1000, dropout_rate=0.1):
        super(FeedForwardNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 128),
            nn.ReLU(),            
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, output_dim),
            ELU(alpha=0.01,beta=0.01)  # ReLU ensures non-negative outputs
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, X_train, y_train, X_val=None, y_val=None, loss='rmse',
                batch_size=8096, epochs=100, learning_rate=0.001, beta=3,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the neural network model.
    
    Args:
        model: FeedForwardNN instance
        X_train: Training features (numpy array)
        y_train: Training targets (numpy array)
        X_val: Validation features (numpy array, optional)
        y_val: Validation targets (numpy array, optional)
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        dict: Training history containing loss values
    """
    
    # Move model to device
    model = model.to(device)
    
    # Create data loaders
    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val is not None and y_val is not None:
        val_dataset = CustomDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if loss == 'rmse':
        criterion = nn.MSELoss()  # MSE loss for RMSE
    else:
        criterion = AsymmetricMSELoss(beta=beta)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [] if X_val is not None else None
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            # Move batch to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = torch.sqrt(criterion(outputs, batch_y))  # RMSE
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if X_val is not None and y_val is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss = torch.sqrt(criterion(outputs, batch_y))  # RMSE
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {avg_train_loss:.4f} - '
                  f'Val Loss: {avg_val_loss:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {avg_train_loss:.4f}')
    
    return history

def predict(model, X, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained FeedForwardNN instance
        X: Input features (numpy array)
        batch_size: Batch size for prediction
        device: Device to use for prediction
    
    Returns:
        numpy.ndarray: Predicted values
    """
    model = model.to(device)
    model.eval()
    
    dataset = CustomDataset(X, np.zeros((len(X), 0)).astype(np.uint8))  # Dummy y values
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    predictions = []
    
    with torch.no_grad():
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
    
    return np.vstack(predictions)
