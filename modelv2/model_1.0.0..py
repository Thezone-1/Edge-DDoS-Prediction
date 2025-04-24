import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from tqdm import tqdm


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
    def __init__(self, file_path, chunk_size, scaler=None, is_training=True, 
                 train_split=0.8, seed=42):
        """
        Custom IterableDataset to load data progressively in chunks.

        :param file_path: Path to the CSV file.
        :param chunk_size: Number of rows to read per chunk.
        :param scaler: Scaler for feature normalization (used during training).
        :param is_training: Whether the dataset is for training or validation.
        :param train_split: Percentage of data to use for training.
        :param seed: Random seed for reproducibility.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.scaler = scaler
        self.is_training = is_training
        self.train_split = train_split
        self.seed = seed
        
        # Get file size to estimate total samples
        self.file_size = os.path.getsize(file_path)
        
        # Set up random state for consistent splitting
        self.random_state = np.random.RandomState(seed)

    def preprocess_chunk(self, chunk):
        """
        Preprocess a chunk of data: normalize features and encode labels.

        :param chunk: DataFrame chunk.
        :return: Processed features and labels as tensors.
        """
        # Handle missing or invalid values
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.fillna(0)
        
        # Extract labels first
        if "Label" in chunk.columns:
            labels = np.where(chunk["Label"].str.lower() == "benign", 0, 1)
            # Ensure only numeric columns are selected, excluding the "Label" column
            features = chunk.drop(columns=["Label"]).select_dtypes(include=[np.number])
        else:
            # If no Label column, assume all columns are features
            features = chunk.select_dtypes(include=[np.number])
            labels = np.zeros(len(features))  # Placeholder labels
        
        # If scaler has feature names, ensure columns match and order is consistent
        if hasattr(self.scaler, 'feature_names_in_'):
            # Reorder columns to match the scaler's expected order
            features = features.reindex(columns=self.scaler.feature_names_in_, fill_value=0)  
            # This ensures all columns are present and in the correct order

        # Convert to numpy array for faster processing
        features = features.values

        # Replace extreme values
        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=np.finfo(np.float32).max,
            neginf=np.finfo(np.float32).min,
        )
        
        # Clip values to prevent issues with scaling
        features = np.clip(features, -1e10, 1e10)

        # Normalize features if scaler is provided
        if self.scaler is not None:
            features = self.scaler.transform(features)

        return torch.FloatTensor(features), torch.LongTensor(labels)

    def __iter__(self):
        """
        Iterator to read the CSV file in chunks and preprocess the data.
        Splits data between training and validation based on random selection.

        :return: Yields batches of features and labels.
        """
        # Set up random state for this iteration
        rng = np.random.RandomState(self.seed)
        
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            # Split chunk into train/val based on random selection
            indices = np.arange(len(chunk))
            rng.shuffle(indices)
            split_idx = int(len(indices) * self.train_split)
            
            # Select appropriate indices based on dataset type
            if self.is_training:
                selected_indices = indices[:split_idx]
            else:
                selected_indices = indices[split_idx:]
                
            # Skip if no indices for this dataset type
            if len(selected_indices) == 0:
                continue
                
            selected_chunk = chunk.iloc[selected_indices]
            features, labels = self.preprocess_chunk(selected_chunk)
            
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

        # Transformer encoder layers (simplified from full transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layer
        self.output_layer = nn.Linear(dim_feedforward, 2)  # Binary classification

    def forward(self, x):
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Project input
        x = self.input_projection(x)
        
        # Reshape for transformer if needed (add sequence dimension)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, features]
            
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling across sequence dimension
        x = x.mean(dim=1) if len(x.shape) > 2 else x
        
        # Classification output
        x = self.output_layer(x)

        return x


# Training function with progress tracking
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10, 
    device=torch.device("cpu"), patience=3
):
    model = model.to(device)
    best_val_loss = float("inf")
    best_model_path = "best_model.pth"
    clip_value = 1.0  # Define a gradient clipping value
    patience_counter = 0
    
    print(f"Training on device: {device}")
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        train_iterator = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=False
        )
        
        # Training loop
        for batch_idx, (batch_features, batch_labels) in enumerate(train_iterator):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

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
            
            # Update progress bar
            if batch_idx % 10 == 0:
                train_iterator.set_postfix({
                    'loss': f'{train_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        # Calculate training metrics
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = 100. * correct / total if total > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        # Use tqdm for validation progress bar
        val_iterator = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
            leave=False
        )

        with torch.no_grad():
            for batch_features, batch_labels in val_iterator:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        # Calculate validation metrics
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        # Calculate additional metrics if we have predictions
        if len(all_preds) > 0 and len(all_labels) > 0:
            precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        else:
            precision = recall = f1 = 0

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs} Results:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print(f"\nTraining complete. Best model saved to '{best_model_path}'.")
    return best_model_path


# Function to prepare data, train the model, and make predictions
def train_with_progressive_loader(file_path="merged_shuffled.csv", chunk_size=10000, 
                                 num_epochs=10, batch_size=32):
    print(f"Starting training process with file: {file_path}")
    print(f"Loading initial data chunk for scaling...")
    
    # Check if MPS (Apple Silicon) is available, otherwise use CUDA or CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the dataset once to fit the scaler
    try:
        data = pd.read_csv(file_path, nrows=chunk_size, low_memory=False)
        print(f"Loaded initial data with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    # Ensure only numeric columns are selected, excluding the "Label" column
    features = data.drop(columns=["Label"], errors="ignore").select_dtypes(include=[np.number])
    print(f"Extracted {features.shape[1]} numeric features for scaling")

    # Handle extreme values before fitting the scaler
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    features_array = np.clip(features.values, -1e10, 1e10)

    # Initialize and fit the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(features_array)
    scaler.feature_names_in_ = features.columns.tolist()  # Store column names explicitly
    
    print(f"Scaler fitted with {len(scaler.feature_names_in_)} features")

    # Create training and validation datasets
    print("Creating progressive datasets...")
    train_dataset = ProgressiveDDoSDataset(
        file_path, chunk_size, scaler=scaler, is_training=True
    )
    val_dataset = ProgressiveDDoSDataset(
        file_path, chunk_size, scaler=scaler, is_training=False
    )

    # Create data loaders with multiple workers for better performance
    print(f"Creating data loaders with batch size {batch_size}...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        num_workers=2,  # Use multiple workers for data loading
        prefetch_factor=2,  # Prefetch batches for better performance
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=2,
    )

    # Initialize model
    input_dim = len(scaler.feature_names_in_)  # Number of features
    print(f"Initializing model with input dimension: {input_dim}")
    model = DDoSTransformer(input_dim=input_dim)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting model training...")
    best_model_path = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
    )
    
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    
    return model, scaler


# Function to predict a single input field
def predict_single(model, scaler, input_field, device=None):
    """
    Predict whether the input field is benign (0) or attack (1).

    :param model: Trained DDoSTransformer model.
    :param scaler: Scaler used for feature normalization.
    :param input_field: List or NumPy array representing a single input field.
    :param device: The device to run inference on.
    :return: Prediction (0 for benign, 1 for attack).
    """
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else \
                 torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if input is a list or numpy array
    if isinstance(input_field, list):
        # Find the label at the end and remove it
        if isinstance(input_field[-1], str):
            input_field = input_field[:-1]
            
        # Convert string items to float where possible
        numeric_input = []
        for x in input_field:
            try:
                numeric_input.append(float(x))
            except (ValueError, TypeError):
                # Skip non-numeric values
                pass
        
        numeric_input = np.array(numeric_input)
    else:
        numeric_input = np.array(input_field)

    # Handle missing or extreme values
    numeric_input = np.nan_to_num(
        numeric_input,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=np.finfo(np.float32).min,
    )
    
    # Clip values to prevent scaling issues
    numeric_input = np.clip(numeric_input, -1e10, 1e10)

    # Ensure we have the expected number of features
    if len(numeric_input) != len(scaler.feature_names_in_):
        print(f"Warning: Input has {len(numeric_input)} features, but the model expects {len(scaler.feature_names_in_)}.")
        # Pad or truncate to match expected size
        if len(numeric_input) < len(scaler.feature_names_in_):
            # Pad with zeros
            padding = np.zeros(len(scaler.feature_names_in_) - len(numeric_input))
            numeric_input = np.concatenate([numeric_input, padding])
        else:
            # Truncate
            numeric_input = numeric_input[:len(scaler.feature_names_in_)]

    # Normalize the input
    input_normalized = scaler.transform(numeric_input.reshape(1, -1))
    input_tensor = torch.FloatTensor(input_normalized).to(device)

    # Perform prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, prediction].item()
        
    return prediction, confidence


if __name__ == "__main__":
    # Set file path - make sure this points to your actual data file
    file_path = "merged_shuffled.csv"
    
    # Check for MPS (Apple Silicon GPU) availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("tpu")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Train the model with progress printing
    model, scaler = train_with_progressive_loader(
        file_path=file_path,
        chunk_size=10000,  # Adjust based on memory constraints
        num_epochs=10,
        batch_size=32
    )

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

    if model is not None and scaler is not None:
        # Predict using the trained model
        prediction, confidence = predict_single(model, scaler, example_input, device)
        print(f"Prediction for input: {prediction} ({'Attack' if prediction == 1 else 'Benign'})")
        print(f"Confidence: {confidence:.4f}")
    else:
        print("Model training failed. Please check the error messages above.")