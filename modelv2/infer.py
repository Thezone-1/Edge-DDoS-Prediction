import torch
import numpy as np
from model import DDoSTransformer  # Import the model architecture
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the scaler used during training
scaler = MinMaxScaler(feature_range=(0, 256))
data = pd.read_csv("merged_shuffled.csv")
X = data.select_dtypes(include=[np.number]).values  # Select only numeric columns

# Replace invalid values in the training data
X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=np.finfo(np.float64).max / 10,  # Scale down large positive values
    neginf=np.finfo(np.float64).min / 10   # Scale up large negative values
)
X = np.clip(X, -1e10, 1e10)  # Clip extreme values

# Fit the scaler on the sanitized training data
scaler.fit(X)

# Define the model architecture
input_dim = X.shape[1]  # Number of input features
model = DDoSTransformer(input_dim=input_dim)

# Load the model weights
model_path = "best_model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Example input datapoint
example_input = ['8', '172.16.0.5-192.168.50.1-60676-80-6', '172.16.0.5', '60676', '192.168.50.1', '80', '6', '2018-12-01 09:17:23.849891', '184', '1', '1', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '10869.565217391304', '184.0', '0.0', '184.0', '184.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0', '0', '0', '0', '32', '32', '5434.782608695652', '5434.782608695652', '0.0', '0.0', '0.0', '0.0', '0.0', '0', '0', '0', '0', '0', '1', '1', '0', '1.0', '0.0', '0.0', '0.0', '32', '0', '0', '0', '0', '0', '0', '1', '0', '1', '0', '64', '227', '0', '32', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0', '1', 'DrDoS_NTP']

# Extract numeric features from the input
numeric_input = np.array([float(x) for x in example_input if x.replace('.', '', 1).isdigit()])

# Ensure the input has the same number of features as the training data
if len(numeric_input) < input_dim:
    # Pad with zeros if the input has fewer features
    numeric_input = np.pad(numeric_input, (0, input_dim - len(numeric_input)), constant_values=0)
elif len(numeric_input) > input_dim:
    # Truncate if the input has more features
    numeric_input = numeric_input[:input_dim]

# Replace invalid values in the input data
numeric_input = np.nan_to_num(
    numeric_input,
    nan=0.0,
    posinf=np.finfo(np.float64).max / 10,  # Scale down large positive values
    neginf=np.finfo(np.float64).min / 10   # Scale up large negative values
)
numeric_input = np.clip(numeric_input, -1e10, 1e10)  # Clip extreme values

# Normalize the numeric input
normalized_input = scaler.transform([numeric_input])

# Convert to a torch tensor
input_tensor = torch.FloatTensor(normalized_input)

# Perform prediction
with torch.no_grad():  # Disable gradient computation for inference
    prediction = model(input_tensor)

# Output the predicted class
predicted_class = prediction.argmax(dim=1).item()
print(f"Prediction: {predicted_class} (0: Benign, 1: Attack)")