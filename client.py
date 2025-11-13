import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import flwr as fl
from model import DistilBERTSentiment
from sklearn.metrics import accuracy_score

class FLClient(fl.client.NumPyClient):
    def __init__(self,model,train_dataset,test_dataset,device):
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset,batch_size = 16, shuffle = True)
        self.test_loader = DataLoader(test_dataset, batch_size = 16)
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _,val in self.model.state_dict().items()]
    
    def set_parameters(self,parameters):
        params_dict = zip(self.model.state_dict().keys(),parameters)
        state_dict = {k: torch.tensor(v) for k,v in params_dict}
        self.model.load_state_dict(state_dict,strict = True)

    def fit(self,parameters,config):
        self.set_parameters(parameters)
        optimizer = AdamW(self.model.parameters(),lr=5e-5)
        self.model.train()
        for batch in self.train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            outputs = self.model(input_ids,attention_mask)
            loss = torch.nn.CrossEntropyLoss()(outputs,labels)
            loss.backward()
            optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs,dim = 1)
                preds.extend(predictions.cpu().numpy())
                targets.extend(labels.cpu().numpy())
            acc = accuracy_score(targets,preds)
            return float(acc), len(self.test_loader.dataset), {"accuracy": float(acc)}