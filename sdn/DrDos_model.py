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
        )  # Binary classification: Normal vs DDoS

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
