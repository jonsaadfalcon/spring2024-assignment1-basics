import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import wandb

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):

        #with open(file_path, 'r', encoding='utf-8') as f:
        #    lines = f.readlines()

        self.examples = []
        lines = ["Hello, world!", "Transformers are models.", "This is a sample dataset."]
        for line in lines:
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
            for i in range(0, len(tokens) - block_size + 1, block_size):  # Overlap is possible
                self.examples.append(torch.tensor(tokens[i:i + block_size], dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def train(model, device, loader, optimizer):
    model.train()
    for epoch in range(3):  # run for more epochs depending on dataset size
        for idx, input_ids in enumerate(loader):
            input_ids = input_ids.to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx % 10 == 0:  # log every 10 batches
                wandb.log({"loss": loss.item()})
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

def main():
    wandb.init(project="LLM_from_Scratch", entity="jonsaadfalcon")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    dataset = TextDataset("path_to_your_dataset.txt", tokenizer)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    train(model, torch.device("cuda"), data_loader, optimizer)

    wandb.finish()

if __name__ == "__main__":
    main()
