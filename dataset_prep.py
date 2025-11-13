from datasets import load_dataset
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
MAX_LENGTH = 128  # max token length

def tokenize(batch):
    return tokenizer(
        batch['sentence'],  # SST-2 column for text
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

def prepare_sst2(num_clients=5, fast_mode=False):
   
    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    
    # Tokenize
    train_dataset = dataset['train'].map(tokenize, batched=True)
    test_dataset = dataset['validation'].map(tokenize, batched=True)

    # Keep only essential columns
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Optional: reduce dataset size for fast testing
    if fast_mode:
        train_dataset = train_dataset.select(range(500))  # first 500 samples
        test_dataset = test_dataset.select(range(200))    # first 200 samples

    # Split train dataset among clients
    client_datasets = []
    client_size = len(train_dataset) // num_clients
    for i in range(num_clients):
        start = i * client_size
        end = start + client_size if i < num_clients - 1 else len(train_dataset)
        client_datasets.append(train_dataset.select(range(start, end)))

    print("Datasets prepared.")
    return client_datasets, test_dataset
