# FedCore: Federated Learning Framework

FedCore is a modular framework for training machine learning models across distributed datasets without moving data from its original location. It implements the Federated Averaging (FedAvg) algorithm and provides a clean API for custom model and client definitions.

## Features

*   **Decentralized Training**: Keep data on-device to ensure privacy.
*   **FedAvg Implementation**: Robust weight aggregation logic.
*   **Modular Architecture**: Easily swap models, datasets, or aggregation strategies.
*   **Extensible Clients**: Support for heterogeneous hardware and local training loops.

## Installation

```bash
pip install -e .
```

## Quick Start

1.  **Define your model**: Inherit from `fedcore.model.FederatedModel`.
2.  **Partition your data**: Use `fedcore.utils.data_manager`.
3.  **Run the server**:

```python
from fedcore.server import Server
from fedcore.client import Client

# ... setup code ...
server = Server(global_model, clients)
server.run_round(1)
```

## Architecture

*   **Server**: Orchestrates the training rounds and maintains the global model state.
*   **Client**: Performs local optimization on private data.
*   **Aggregator**: Merges client updates using secure or weighted averaging.

## Security

This framework is designed for research. For production use, consider adding:
*   Differential Privacy (DP)
*   Secure Multi-Party Computation (SMPC)
*   Homomorphic Encryption

## License

MIT
