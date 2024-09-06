import pandas as pd 
import torch 
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils.data_preprocessing import text_preprocessing


class SentimentDataset(Dataset):
    ''' # Custom Class for BERT '''
    def __init__(self, texts, labels=None, tokenizer=None, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return input_ids, attention_mask, label
        else:
            return input_ids, attention_mask


# Function to perform sentiment analysis using BERT
def perform_sa(data, content_col, label_col=None, pretrained_model='bert-base-uncased', batch_size=16, epochs=3):
    '''Main code that performs Sentiment Analysis using BERT'''

    # Preprocess the content text
    data = data.dropna(subset=[content_col])
    data['content'] = data[content_col].apply(text_preprocessing)

    # Load the pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=3)  # Assuming 3 sentiment classes: positive, neutral, negative

    # Prepare the dataset
    X_train, X_val, y_train, y_val = train_test_split(data['content'], data[label_col], test_size=0.2, random_state=42)
    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_dataset = SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # Validation loop
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())

    # Evaluate performance
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['negative', 'neutral', 'positive'])
    
    print(f'Validation Accuracy: {accuracy}')
    print(report)
