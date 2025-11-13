import torch
import flwr as fl
from model import DistilBERTSentiment

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            model = DistilBERTSentiment()
            weights = fl.common.parameters_to_ndarrays(aggregated_weights)
            params_dict = zip(model.state_dict().keys(), weights)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            torch.save(model.state_dict(),'global_model.pth')
            print(f'✅ Global model saved after round {rnd} -> global_model.pth')
        return aggregated_weights

def start_server(num_rounds=3):
    strategy = SaveModelStrategy()
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
        