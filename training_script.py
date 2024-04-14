import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import wandb

from tests.tokenizer import Tokenizer
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        
        self.examples = []

        print("Loading training text!")
        with open(file_path, 'rb') as f:
    
            for line in tqdm(f.readlines()[1000:1010]):
                
                #print("Line:", line.decode('utf-8'))
                line_in_bytes = line
                line_decoded = line_in_bytes.decode('utf-8')
                tokens = tokenizer.encode(line_decoded)

                for i in range(0, max(block_size, len(tokens) - block_size + 1), block_size):
                    self.examples.append(torch.tensor(tokens[i:i + block_size], dtype=torch.long))

        ##########################################################

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

    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    dataset = TextDataset(file_path, tokenizer)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train(model, torch.device("cuda"), data_loader, optimizer)

    wandb.finish()

if __name__ == "__main__":
    main()
