# Federated Learning-based Sentiment Analysis using DistilBERT

## 🧠 Overview
This project demonstrates a **privacy-preserving Federated Learning (FL)** framework for **sentiment analysis** using **DistilBERT**, a transformer-based NLP model.  
The system classifies movie reviews as **positive** or **negative**, with decentralized training ensuring no raw data leaves the clients.

---

## 🚀 Features
- Federated Learning simulation with **5 clients**
- **DistilBERT** model fine-tuned for sentiment classification
- **FedAvg** aggregation for global model updates
- **Flask API** for backend inference
- **Streamlit** web interface for real-time predictions
- End-to-end pipeline from data preprocessing to deployment

---

## 📂 Project Structure

federated-learning-sentiment-analysis/
│
├── dataset_prep.py # Loads and partitions SST-2 dataset
├── model.py # Defines DistilBERTSentiment model architecture
├── client.py # Client class for local training & evaluation
├── main.py # Main federated learning loop (training & aggregation)
├── server.py # Optional FL server setup
├── app.py # Flask API for sentiment prediction
├── app1.py # Streamlit interface for real-time interaction
├── global_model.pth # Trained global model
└── README.md # Project documentation


---

## 🧩 Model Architecture
- **Base Model:** DistilBERT (`distilbert-base-uncased`)
- **Layers:**
  - Pre-classifier (Linear + ReLU)
  - Dropout (0.3)
  - Output layer (2 classes: Positive, Negative)
- **Optimizer:** Adam
- **Learning Rate:** 2e-5
- **Rounds:** 3 federated rounds

---

## 🧾 Dataset
**Stanford Sentiment Treebank (SST-2)**  
- Task: Binary sentiment classification  
- Size: 67,000+ labeled sentences  
- Distribution: 50% positive, 50% negative  
- Split into 5 clients for federated simulation

---

## ⚙️ Federated Learning Workflow
1. Initialize global model
2. Distribute global weights to all clients
3. Clients train locally and send updated weights
4. Server aggregates using **FedAvg**
5. Evaluate and redistribute updated global model

---

## 📊 Results
| Round | Accuracy | Loss  |
|:------|:----------|:------|
| 1     | 90.8%     | 0.87  |
| 2     | 90.4%     | 0.87  |
| 3     | 91.2%     | 0.86  |

**Final Accuracy:** ~91.2%

---

## 🖥️ Deployment

### Flask API
Run the Flask backend:
```bash
python app.py
