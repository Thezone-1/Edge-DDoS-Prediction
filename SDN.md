# Network Topology and Controller setup

## Overview
The setup includes a custom Python-based OpenFlow switch application (`Switch`) and a network topology script (`network-topology.py`) using **Ryu** controller and **Mininet**..

<img width="770" alt="Screenshot 2024-11-22 at 14 16 36" src="https://github.com/user-attachments/assets/4561dc35-7f9e-4047-9390-785521e76fa2">

---

## Components

### 1. **Ryu Controller Application** (`topology.py`)
A Ryu-based controller application that implements an OpenFlow switch with basic learning capabilities.

- **Flow Rules**: Dynamically installs flows to handle packet forwarding.
- **MAC Learning**: Learns MAC-to-port mappings to avoid flooding.
- **Packet Handling**:
  - Processes incoming packets (`_packet_in_handler`).
  - WIP: Placeholder for integrating machine learning model.

#### Key Methods:
- `add_flow`: Adds flow entries to the switch.
- `_packet_in_handler`: Handles packets sent to the controller, performs MAC learning, and forwards packets accordingly.

---

### 2. **Network Topology Script** (`network-topology.py`)
Defines a simple network topology using **Mininet**.

#### Topology Details:
- **Hosts**: `h1`, `h2`, `h3`
- **Switch**: `s1`
- **Controller**: Remote Ryu controller at `127.0.0.1:6653`.

#### Workflow:
1. Adds hosts and connects them to a single switch.
2. Starts the Mininet CLI for manual interaction.
3. Connects the topology to the Ryu controller.

---

## Setup and Execution

### Prerequisites
- **Ryu** SDN framework.
- **Mininet** network emulator.
- Python 3 environment.

### Steps:
1. **Run Ryu Controller**:
   ```bash
   ryu-manager topology.py
   ```
2. **Start Mininet Topology**:
   ```bash
    python3 network-topology.py
   ```
3. **Interact with CLI**:
   - Use Mininet CLI commands to test connectivity (e.g., `pingall`).
   - Add traffic or test scenarios for predictive logic.

---

## Future Work
- **Integrate Machine Learning**: Extend the `_packet_in_handler` to classify packets using a pre-trained ML model.
- **Advanced Topologies**: Build more complex topologies for performance testing.

This setup provides a foundational SDN environment for further experimentation with the project.
