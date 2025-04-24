import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# Custom Dataset class for DDoS data
class DDoSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Progressive Dataset class for DDoS data
class ProgressiveDDoSDataset(IterableDataset):
    def __init__(self, file_path, chunk_size, scaler=None, is_training=True):
        """
        Custom IterableDataset to load data progressively in chunks.

        :param file_path: Path to the CSV file.
        :param chunk_size: Number of rows to read per chunk.
        :param scaler: Scaler for feature normalization (used during training).
        :param is_training: Whether the dataset is for training or validation.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.scaler = scaler
        self.is_training = is_training

    def preprocess_chunk(self, chunk):
        """
        Preprocess a chunk of data: normalize features and encode labels.

        :param chunk: DataFrame chunk.
        :return: Processed features and labels as tensors.
        """
        # Ensure only numeric columns are selected, excluding the "Label" column
        if "Label" in chunk.columns:
            features = chunk.drop(columns=["Label"]).select_dtypes(include=[np.number])
        else:
            features = chunk.select_dtypes(include=[np.number])

        # Ensure the columns match the scaler's expectations
        features = features[self.scaler.feature_names_in_]

        # Replace infinite values with NaN and handle them
        features = np.nan_to_num(
            features.values,
            nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=np.finfo(np.float64).min,
        )

        # Normalize features if scaler is provided
        if self.scaler:
            if features.shape[1] != self.scaler.n_features_in_:
                raise ValueError(
                    f"Chunk has {features.shape[1]} features, but the scaler expects {self.scaler.n_features_in_} features."
                )
            features = self.scaler.transform(features)

        # Encode labels: "benign" -> 0, others -> 1
        labels = np.where(chunk["Label"].str.lower() == "benign", 0, 1)

        return torch.FloatTensor(features), torch.LongTensor(labels)

    def __iter__(self):
        """
        Iterator to read the CSV file in chunks and preprocess the data.

        :return: Yields batches of features and labels.
        """
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            features, labels = self.preprocess_chunk(chunk)
            for feature, label in zip(features, labels):
                yield feature, label


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

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=dim_feedforward,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            batch_first=True,
        )

        # Simplified output layer
        self.output_layer = nn.Linear(
            dim_feedforward, 2
        )  # Binary classification: Benign vs Attack

    def forward(self, x):
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create a dummy target tensor for transformer (same shape as x)
        tgt = torch.zeros_like(x)  # Dummy target with same shape

        # Transform using the transformer model
        x = self.transformer(x, tgt)

        # Get classification output
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
    clip_value = 1.0  # Define a gradient clipping value

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
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
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
        all_preds = []
        all_labels = []

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

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        # Calculate additional metrics
        precision = precision_score(all_labels, all_preds, average="binary")
        recall = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(
            f"Train Loss: {train_loss/len(train_loader)}, Accuracy: {100.*correct/total}%"
        )
        print(
            f"Val Loss: {val_loss/len(val_loader)}, Accuracy: {100.*val_correct/val_total}%"
        )
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss}")

    print("Training complete. Best model saved to 'best_model.pth'.")
    return best_model_path


# Function to prepare data, train the model, and make predictions
def train_with_progressive_loader():
    file_path = "merged_shuffled.csv"
    chunk_size = 10000  # Adjust based on memory constraints

    # Load the dataset once to fit the scaler
    data = pd.read_csv(file_path, nrows=chunk_size * 10, low_memory=False)  # Use a subset for scaling
    # Ensure only numeric columns are selected, excluding the "Label" column
    features = data.drop(columns=["Label"], errors="ignore").select_dtypes(include=[np.number])

    # Clip feature values to a specific range before fitting the scaler
    features = np.clip(features, -1e10, 1e10)

    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(0, 256))

    # Fit the scaler on the DataFrame and store column names
    scaler.fit(features)
    scaler.feature_names_in_ = features.columns  # Store column names explicitly

    # Create training and validation datasets
    train_dataset = ProgressiveDDoSDataset(file_path, chunk_size, scaler=scaler)
    val_dataset = ProgressiveDDoSDataset(file_path, chunk_size, scaler=scaler, is_training=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model
    input_dim = features.shape[1]  # Number of features
    model = DDoSTransformer(input_dim=input_dim)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    device = torch.device("mps")
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=10,
        device=device,
    )

    return model, scaler


# Ensure the device is set to "mps" for macOS GPU
device = torch.device("mps")

# Function to predict a single input field
def predict_single(model, scaler, input_field):
    """
    Predict whether the input field is benign (0) or attack (1).

    :param model: Trained DDoSTransformer model.
    :param scaler: Scaler used for feature normalization.
    :param input_field: List or NumPy array representing a single input field.
    :return: Prediction (0 for benign, 1 for attack).
    """
    # Remove the last element (Label) from the input field
    input_field = input_field[:-1]

    # Filter numeric features from the input
    numeric_input = np.array([float(x) for x in input_field if x.replace('.', '', 1).isdigit()])

    # Replace NaN and infinity values with finite numbers
    numeric_input = np.nan_to_num(
        numeric_input,
        nan=0.0,  # Replace NaN with 0.0
        posinf=np.finfo(np.float32).max,  # Replace positive infinity with max float
        neginf=np.finfo(np.float32).min,  # Replace negative infinity with min float
    )

    # Clip values to a reasonable range to avoid overflow
    numeric_input = np.clip(numeric_input, -1e10, 1e10)

    # Ensure the input has the same number of features as the scaler expects
    if len(numeric_input) != scaler.n_features_in_:
        raise ValueError(f"Input has {len(numeric_input)} features, but the scaler expects {scaler.n_features_in_} features.")

    # Normalize the numeric input
    input_tensor = torch.FloatTensor(scaler.transform([numeric_input])).to(device)

    # Perform prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
    return prediction


if __name__ == "__main__":
    model, scaler = train_with_progressive_loader()

    # Move the model to the GPU
    model = model.to(device)

    # Example usage of prediction
    example_input = [
        '7961', '192.168.50.7-4.2.2.4-57057-53-17', '192.168.50.7', '57057', '4.2.2.4',
        '53', '17', '2018-12-01 09:17:30.332888', '20604', '2', '2', '86.0', '118.0',
        '43.0', '43.0', '43.0', '0.0', '59.0', '59.0', '59.0', '0.0', '9900.9900990099',
        '194.1370607649', '6868.0', '11890.528836010615', '20598.0', '2.0', '4.0', '4.0',
        '0.0', '4.0', '4.0', '2.0', '2.0', '0.0', '2.0', '2.0', '0', '0', '0', '0', '64',
        '40', '97.06853038245', '97.06853038245', '43.0', '59.0', '49.4', '8.763560920082657',
        '76.8', '0', '0', '0', '0', '0', '0', '0', '0', '1.0', '61.75', '43.0', '59.0',
        '64', '0', '0', '0', '0', '0', '0', '2', '86', '2', '118', '-1', '-1', '1', '32',
        '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0', '0', 'BENIGN'
    ]

    # Predict using the trained model
    prediction = predict_single(model, scaler, example_input)
    print(f"Prediction for input: {prediction}")