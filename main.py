import torch
from dataset_prep import prepare_sst2
from client import FLClient
from model import DistilBERTSentiment

# ------------------------
# Configuration
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = 5
num_rounds = 3  # Number of federated learning rounds

# ------------------------
# Prepare datasets
# ------------------------
print("Preparing datasets...")
client_datasets, test_dataset = prepare_sst2(num_clients)
print(f"Prepared datasets for {num_clients} clients.\n")

# ------------------------
# Initialize clients
# ------------------------
clients = []
for i in range(num_clients):
    model = DistilBERTSentiment().to(device)
    client = FLClient(model, client_datasets[i], test_dataset, device)
    clients.append(client)
print(f"Initialized {num_clients} clients.\n")
# ------------------------
# Federated Learning Rounds
# ------------------------
for round_num in range(1, num_rounds + 1):
    print(f"====== Round {round_num} ======")

    # Step 1: Get global weights from the first client
    global_weights = clients[0].get_parameters(config={})

    # Step 2: Train each client on their local data
    for i, client in enumerate(clients):
        print(f"Training Client {i}...")
        client.fit(global_weights, config={})
    
    # Step 3: Aggregate updated weights (simple averaging)
    new_global_weights = []
    for param_idx in range(len(global_weights)):
        param_sum = sum(client.get_parameters(config={})[param_idx] for client in clients)
        new_global_weights.append(param_sum / num_clients)
    
    # Update global weights
    global_weights = new_global_weights

    # Step 4: Evaluate each client
    for i, client in enumerate(clients):
        acc, loss, _ = client.evaluate(global_weights, config={})
        print(f"Client {i} Accuracy: {acc:.4f}, Loss: {loss:.4f}")
    
    print(f"====== End of Round {round_num} ======\n")

# ------------------------
# Save the final global model
# ------------------------
print("Saving the final global model...")
model = DistilBERTSentiment()
state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), global_weights)}
model.load_state_dict(state_dict, strict=True)

save_path = "/kaggle/working/global_model.pth"
torch.save(model.state_dict(), save_path)
print(f"✅ Global model saved at {save_path}")

print("Federated learning completed!")


