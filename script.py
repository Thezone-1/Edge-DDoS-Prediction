import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scapy.all import sniff, IP
import logging
from datetime import datetime
import threading
import queue
import socket
import json
import os


class DDoSTransformer(nn.Module):
    def __init__(
        self, input_dim, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1
    ):
        super(DDoSTransformer, self).__init__()

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

        x = self.input_projection(x)

        x = self.pos_encoder(x)


        x = self.transformer_encoder(x)

        x = x.mean(dim=1) if len(x.shape) > 2 else x
        x = self.output_layer(x)
        return x

class DDoSStreamClassifier:
    def __init__(self, model_path, feature_extractor, max_queue_size=1000, log_path='ddos_detection.log'):
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )


        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.model = DDoSTransformer(input_dim=41)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.feature_extractor = feature_extractor


        self.packet_queue = queue.Queue(maxsize=max_queue_size)


        self.attack_threshold = 0.7
        self.window_size = 100

        self.mitigation_callback = None

    def extract_network_features(self, packet):
        
        try:
            features = [
                packet[IP].src,  # Source
                packet[IP].dst,  # Destination
                packet[IP].proto,  # Protocol
                len(packet),  # Packet length
            ]
            return np.array(features)
        except Exception as e:
            logging.error(f"Feature extraction error: {e}")
            return None

    def packet_handler(self, packet):
        
        if IP in packet:
            try:

                features = self.extract_network_features(packet)

                if features is not None:
                    if not self.packet_queue.full():
                        self.packet_queue.put(features)
            except Exception as e:
                logging.error(f"Packet handling error: {e}")

    def classify_stream(self):
        # CONTINOUSLY CLASSIFY 
        packet_window = []

        while True:
            try:

                while len(packet_window) < self.window_size:
                    packet = self.packet_queue.get(timeout=5)
                    packet_window.append(packet)


                X = torch.FloatTensor(packet_window).to(self.device)

                with torch.no_grad():
                    outputs = self.model(X)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)

                ddos_prob = probabilities[:, 1].mean().item()
                is_attack = ddos_prob > self.attack_threshold

                if is_attack:
                    logging.warning(f"Potential DDoS detected! Probability: {ddos_prob:.2%}")


                    if self.mitigation_callback:
                        self.mitigation_callback(packet_window)

                packet_window.clear()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Stream classification error: {e}")

    def start_monitoring(self, interface='eth0'):
        
        classification_thread = threading.Thread(target=self.classify_stream)
        classification_thread.daemon = True
        classification_thread.start()

        print(f"Starting packet capture on {interface}")
        sniff(iface=interface, prn=self.packet_handler, store=0)

    def set_mitigation_callback(self, callback):
        
        self.mitigation_callback = callback

def simple_mitigation_callback(attack_packets):
    
    source_ips = set(packet[0] for packet in attack_packets)

    for ip in source_ips:
        os.system(f"sudo iptables -A INPUT -s {ip} -j DROP")
        logging.warning(f"Blocked potential attack source: {ip}")

def main():
    # Path to your saved PyTorch model
    MODEL_PATH = 'best_model.pth'

    # Initialize classifier
    classifier = DDoSStreamClassifier(
        model_path=MODEL_PATH,
        feature_extractor=None
    )

    # Optional: Set mitigation callback
    classifier.set_mitigation_callback(simple_mitigation_callback)

    classifier.start_monitoring(interface='eth0')

if __name__ == "__main__":
    main()