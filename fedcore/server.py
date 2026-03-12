import logging
from fedcore.aggregator import Aggregator

class Server:
    """
    The central coordinator for the federated learning process.
    """
    def __init__(self, model, clients):
        self.global_model = model
        self.clients = clients
        self.logger = logging.getLogger("Server")

    def run_round(self, round_idx, local_epochs=1):
        self.logger.info(f"Starting Round {round_idx}")
        client_updates = []
        client_sizes = []

        for client in self.clients:
            # Send global model to client
            local_model = copy.deepcopy(self.global_model)
            weights, size = client.train(local_model, epochs=local_epochs)
            client_updates.append(weights)
            client_sizes.append(size)

        # Aggregate updates
        new_weights = Aggregator.aggregate(client_updates, client_sizes)
        self.global_model.set_weights(new_weights)
        self.logger.info(f"Round {round_idx} complete.")

import copy
