from fedcore.utils.logger import setup_logging
from fedcore.utils.data_manager import get_mnist_partitions
from fedcore.client import Client
from fedcore.server import Server
from fedcore.model import FederatedModel

def main():
    setup_logging()
    
    # Configuration
    NUM_CLIENTS = 3
    ROUNDS = 5
    LOCAL_EPOCHS = 2
    
    # Data partitioning
    partitions = get_mnist_partitions(NUM_CLIENTS)
    
    # Initialize clients
    clients = [Client(i, partitions[i]) for i in range(NUM_CLIENTS)]
    
    # Initialize server and global model
    global_model = FederatedModel()
    server = Server(global_model, clients)
    
    # Federated Training Loop
    for r in range(1, ROUNDS + 1):
        server.run_round(r, local_epochs=LOCAL_EPOCHS)
        
    print("Federated Training Finished Successfully.")

if __name__ == "__main__":
    main()
