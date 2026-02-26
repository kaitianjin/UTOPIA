import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

## This version accepts the label y as numerical integer values from 0 to num_cell_types,
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CrossEntropyLoss(nn.Module):
    def __init__(self, n_classes, label_smoothing=0):
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes
        
    def forward(self, outputs, targets):
        # Convert targets to one-hot encoding
        one_hot_targets = torch.zeros(targets.size(0), self.n_classes).cuda().scatter_(1, targets.view(-1, 1), 1)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            smooth_target = one_hot_targets * (1 - self.label_smoothing) + self.label_smoothing / self.n_classes
            one_hot_targets = smooth_target
            
        # Compute cross entropy loss with one-hot targets
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -torch.sum(one_hot_targets * log_probs, dim=1)
        return loss.mean()
    
class FeedForwardClassifier(nn.Module):
    def __init__(self, input_dim=200, output_dim=10, dropout_rate=0.3, temperature=1.0):
        super(FeedForwardClassifier, self).__init__()
        
        self.temperature = temperature 
        self.output_dim = output_dim
        
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
            
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        logits = self.network(x)
        # Apply temperature scaling to logits before softmax
        scaled_logits = logits / self.temperature
        return nn.functional.softmax(scaled_logits, dim=1)

def train_model(model, X_train, y_train, X_val=None, y_val=None,
                batch_size=32, epochs=100, learning_rate=0.001, weight_decay=1e-4,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the neural network classification model with regularization.
    
    Args:
        model: FeedForwardClassifier instance
        X_train: Training features (numpy array)
        y_train: Training soft labels (numpy array)
        X_val: Validation features (numpy array, optional)
        y_val: Validation soft labels (numpy array, optional)
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization strength (weight decay parameter)
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        dict: Training history containing loss and accuracy values
    """
    
    # Move model to device
    model = model.to(device)
    
    # Create data loaders
    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    if X_val is not None and y_val is not None:
        val_dataset = CustomDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Label smoothing loss - helps prevent overconfidence
    criterion = CrossEntropyLoss(n_classes=model.output_dim, label_smoothing=0)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [] if X_val is not None else None,
        'val_acc': [] if X_val is not None else None
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_accs = []
        
        for batch_X, batch_y in train_loader:
            # Move batch to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            
            # Get logits before softmax for loss calculation
            logits = model.network(batch_X)
            logits = logits / model.temperature
            
            # Calculate loss with the logits
            loss = criterion(logits, batch_y)
            
            # Calculate accuracy
            _, pred_indices = torch.max(outputs, 1)
            true_indices = batch_y
            hard_accuracy = (pred_indices == true_indices).float().mean().item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            train_accs.append(hard_accuracy)
        
        # Calculate average training metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        
        # Validation
        if X_val is not None and y_val is not None:
            model.eval()
            val_losses = []
            val_accs = []
            val_confidences = []  # Track prediction confidence
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # Get logits for loss calculation
                    logits = model.network(batch_X)
                    logits = logits / model.temperature
                    
                    # Get softmax outputs for accuracy
                    outputs = nn.functional.softmax(logits, dim=1)
                    
                    val_loss = criterion(logits, batch_y)
                    
                    # Calculate accuracy
                    _, pred_indices = torch.max(outputs, 1)
                    true_indices = batch_y
                    hard_accuracy = (pred_indices == true_indices).float().mean().item()
                    
                    # Track prediction confidence (max probability)
                    confidence = torch.max(outputs, dim=1)[0].mean().item()
                    val_confidences.append(confidence)
                    
                    val_losses.append(val_loss.item())
                    val_accs.append(hard_accuracy)
            
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accs)
            avg_confidence = np.mean(val_confidences)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(avg_val_acc)
            
            # print(f'Epoch [{epoch+1}/{epochs}] - '
            #       f'Train Loss: {avg_train_loss:.4f} - '
            #       f'Train Acc: {avg_train_acc:.4f} - '
            #       f'Val Loss: {avg_val_loss:.4f} - '
            #       f'Val Acc: {avg_val_acc:.4f} - '
            #       f'Avg Confidence: {avg_confidence:.4f}')
        # else:
            # print(f'Epoch [{epoch+1}/{epochs}] - '
            #       f'Train Loss: {avg_train_loss:.4f} - '
            #       f'Train Acc: {avg_train_acc:.4f}')
    
    return history

def calibrate_temperature(model, X_val, y_val, batch_size=32,
                         device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calibrate the temperature parameter using validation data.
    
    Args:
        model: Trained FeedForwardClassifier instance
        X_val: Validation features (numpy array)
        y_val: Validation soft labels (numpy array)
        batch_size: Batch size for calibration
        device: Device to use for calibration
    
    Returns:
        float: Optimal temperature value
    """
    model = model.to(device)
    model.eval()
    
    # Create validation dataset and loader
    val_dataset = CustomDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define temperature as a learnable parameter
    temperature = nn.Parameter(torch.ones(1).to(device))
    
    # Use NLL loss for temperature scaling
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer for temperature
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
    
    def eval_loss():
        loss = 0
        count = 0
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            with torch.no_grad():
                logits = model.network(batch_X)
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Calculate NLL loss
            batch_loss = criterion(scaled_logits, batch_y)
            loss += batch_loss * batch_X.size(0)
            count += batch_X.size(0)
        
        loss = loss / count
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    # Return optimal temperature
    return temperature.item()

def predict(model, X, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained FeedForwardClassifier instance
        X: Input features (numpy array)
        batch_size: Batch size for prediction
        device: Device to use for prediction
    
    Returns:
        numpy.ndarray: Predicted probability distributions for each class
    """
    model = model.to(device)
    model.eval()
    
    dataset = CustomDataset(X, np.zeros((len(X), 10)))  # Dummy y values
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    predictions = []
    
    with torch.no_grad():
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
    
    return np.vstack(predictions)

