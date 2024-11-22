# Edge-DDos-prediction
DDos prediction using Transformers in edge computing

## Transformer-Based DDoS Detection Model Documentation

### Overview
This project implements a transformer-based neural network to detect DDoS attacks. It involves feature scaling, dataset creation, model training, and validation. The transformer leverages positional encoding and multi-head attention for feature learning.

---

### Dataset
**`DDoSDataset`**: Custom PyTorch dataset for feature-label pairs.  
- **Methods**:
  - `__len__`: Returns dataset size.
  - `__getitem__`: Retrieves a feature-label pair by index.

---

### Model Architecture
**`DDoSTransformer`**: A binary classifier with transformer encoder layers.  

- **Components**:
  1. **Input Projection**: Maps features to higher dimensions.
  2. **Positional Encoding**: Adds sequential information.
  3. **Transformer Encoder**: Multi-head attention layers with feedforward networks.
  4. **Output Layer**: Reduces features to class probabilities.

- **Parameters**:
  - `input_dim`: Number of features.
  - `num_heads`: Attention heads (default: 4).
  - `num_layers`: Encoder layers (default: 2).
  - `dim_feedforward`: Hidden layer size (default: 128).

---

### Training
**`train_model`**: Trains and validates the model, saving the best version.  
- **Input**: Model, dataloaders, loss function, optimizer, epochs, and device.
- **Workflow**:
  1. Train using batches, calculate loss, and update weights.
  2. Validate at each epoch, saving the model with the lowest validation loss.

---

### Example Usage
```python
# Data preparation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = DDoSDataset(X_train, y_train)
test_dataset = DDoSDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model and training
model = DDoSTransformer(input_dim=X_train.shape[1])
train_model(
    model, train_loader, test_loader, nn.CrossEntropyLoss(),
    optim.Adam(model.parameters()), num_epochs=100, device="cuda"
)
```

---

### Key Hyperparameters
| **Parameter**      | **Default** | **Description**                    |
|---------------------|-------------|------------------------------------|
| `num_heads`        | 4           | Number of attention heads.         |
| `num_layers`       | 2           | Number of encoder layers.          |
| `dim_feedforward`  | 128         | Hidden layer size.                 |
| `dropout`          | 0.1         | Dropout rate.                      |
| `batch_size`       | 32          | Samples per batch.                 |
| `learning_rate`    | 0.01        | Optimizer learning rate.           |
| `num_epochs`       | 100         | Total training epochs.             |

---

### Outputs
- **Metrics**: Training and validation loss/accuracy per epoch.
- **Saved Model**: Best model stored as `best_model.pth`.

To load:
```python
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```

---

### Applications
- Detects DDoS attacks in network traffic.
- Adaptable for multi-class anomaly detection.
- Scalable for edge device security.

---

### Enhancements
- Address dataset imbalance with augmentation.
- Optimize hyperparameters with tools like Optuna.
- Visualize attention weights for model interpretability.

This concise documentation is designed for easy integration into a project description or user manual.
