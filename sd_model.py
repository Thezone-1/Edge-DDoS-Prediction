import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Custom Dataset class for APPA DDoS data
class DDoSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Transformer Encoder for DDoS Detection
class DDoSTransformer(nn.Module):
    def __init__(
        self, input_dim, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1
    ):
        super(DDoSTransformer, self).__init__()

        # Embedding layer to project input features
        self.input_projection = nn.Linear(input_dim, dim_feedforward)

        # Positional encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward), nn.ReLU(), nn.Dropout(dropout)
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(dim_feedforward, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # Binary classification: Normal vs DDoS
        )

    def forward(self, x):
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transform
        x = self.transformer_encoder(x)

        # Get classification output
        # Use mean pooling over the sequence
        x = x.mean(dim=1) if len(x.shape) > 2 else x
        x = self.output_layer(x)
        return x


# Training function
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cuda"
):
    model = model.to(device)
    best_val_loss = float("inf")
    best_model_path = "best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(
                device
            )

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(
                    device
                ), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(
            f"Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%"
        )
        print(
            f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100.*val_correct/val_total:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    print("Training complete. Best model saved to 'best_model.pth'.")
    return best_model_path


# Example usage
def prepare_and_train():
    # Assuming you have your data loaded as X (features) and y (labels)
    # Replace this with your actual data loading code
    # Example:
    data = pd.read_csv("APA-DDoS-Dataset.csv")
    X = data.drop("Label", axis=1).values
    y = data["Label"].values

    # For demonstration, creating dummy data
    X = np.random.randn(1000, 41)  # Adjust feature dimension based on your dataset
    y = np.random.randint(0, 2, 1000)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create datasets
    train_dataset = DDoSDataset(X_train, y_train)
    test_dataset = DDoSDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]  # Number of features
    model = DDoSTransformer(input_dim=input_dim)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=1000,
        device=device,
    )

    return model, scaler


if __name__ == "__main__":
    model, scaler = prepare_and_train()
