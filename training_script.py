import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import wandb

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)  # Assuming MLM task
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            wandb.log({"loss": loss.item()})
            print(f'Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item()}')

def main():
    # Initialize Weights & Biases
    wandb.init(project='LLM_from_Scratch', entity='jonsaadfalcon')

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)

    # Example data (replace with your actual dataset)
    texts = ["Hello, world!", "Transformers are models.", "This is a sample dataset."]
    dataset = TextDataset(texts, tokenizer)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch)

    # End wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
