import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel #, AdamW
import wandb
from tqdm import tqdm

from tests.tokenizer import Tokenizer
from tests.Transformer import Transformer_LM
from tests.optimizer import AdamW

##########################################################

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        
        self.examples = []

        print("Loading training text!")
        with open(file_path, 'rb') as f:
    
            for line in tqdm(f.readlines()[1000:1003]):
                
                #print("Line:", line.decode('utf-8'))
                line_in_bytes = line
                line_decoded = line_in_bytes.decode('utf-8')
                tokens = tokenizer.encode(line_decoded)

                #breakpoint()

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

    model_config = {
        "name:": "testing_transformer",
        "vocab_size": 10000,
        "context_length": 32,
        "num_layers": 6,
        "d_model": 400,
        "num_heads": 6,
        "d_ff": 1600,
        "attn_pdrop": 0.1,
        "residual_pdrop": 0.1,
        "weights": torch.load("tests/fixtures/transformer_lm_weights.pt"),
        "save_path": "transformer_saved/transformer_lm_weights.pt"
    }

    #########################

    tokenizer = Tokenizer.from_files(vocab_filepath='tokenizer_saved/ts_vocab.txt',
                                     merges_filepath='tokenizer_saved/ts_merges.txt', 
                                     special_tokens=["|endoftext|"])

    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    dataset = TextDataset(file_path, tokenizer)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    #########################

    #print("dataset examples:" + str(dataset.examples))

    #model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    model =  Transformer_LM(vocab_size = model_config["vocab_size"],
                            context_length = model_config["context_length"],
                            d_model = model_config["d_model"],
                            num_layers = model_config["num_layers"],
                            num_heads = model_config["num_heads"],
                            d_ff = model_config["d_ff"],
                            attn_pdrop = model_config["attn_pdrop"],
                            residual_pdrop = model_config["residual_pdrop"],
                            weights = model_config["weights"],)
    #optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train(model, torch.device("cuda"), data_loader, optimizer)

    wandb.finish()

if __name__ == "__main__":
    main()
