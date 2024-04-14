import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel #, AdamW
import wandb
from tqdm import tqdm

from tests.tokenizer import Tokenizer
from tests.Transformer import Transformer_LM
from tests.optimizer import AdamW, gradient_clipping, get_lr_cosine_schedule

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

def train(model, device, loader, optimizer, learning_scheduler_config, epochs=3):
    model.train()
    for epoch in range(epochs):  # run for more epochs depending on dataset size
        for idx, input_ids in enumerate(loader):
            
            input_ids = input_ids.to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            gradient_clipping(model.parameters(), max_l2_norm=1.0)

            breakpoint()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            optimizer.defaults["lr"] = get_lr_cosine_schedule(it=idx, 
                                                              max_learning_rate=learning_scheduler_config['max_learning_rate'],
                                                              min_learning_rate=learning_scheduler_config['min_learning_rate'],
                                                              warmup_iters=learning_scheduler_config['warmup_iters'],
                                                              cosine_cycle_iters=learning_scheduler_config['cosine_cycle_iters'])

            if idx % 10 == 0:  # log every 10 batches
                wandb.log({"loss": loss.item()})
                print(f"Epoch: {epoch}, Loss: {loss.item()}, LR: {optimizer.defaults['lr']}")

def main():
    wandb.init(project="LLM_from_Scratch", entity="jonsaadfalcon")

    model_config = {
        "name:": "testing_transformer",
        "vocab_size": 10000,
        "context_length": 64, #1024
        "num_layers": 36,
        "d_model": 1280,
        "num_heads": 20,
        "d_ff": 6400,
        "attn_pdrop": 0.1,
        "residual_pdrop": 0.1,
        "weights": None, #torch.load("tests/fixtures/transformer_lm_weights.pt"),
        "save_path": "transformer_saved/transformer_lm_weights.pt"
    }

    learning_scheduler_config = {
        "max_learning_rate": 5e-5,
        "min_learning_rate": 1e-5,
        "warmup_iters": 1000,
        "cosine_cycle_iters": 10000
    }

    ##################################################

    tokenizer = Tokenizer.from_files(vocab_filepath='tokenizer_saved/ts_vocab.txt',
                                     merges_filepath='tokenizer_saved/ts_merges.txt', 
                                     special_tokens=["|endoftext|"])

    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    dataset = TextDataset(file_path, tokenizer, block_size=model_config["context_length"])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    ##################################################

    #print("dataset examples:" + str(dataset.examples))

    device = torch.device("cuda:0")

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
    model.to(device)
    
    ##################################################
    
    #optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer = AdamW(model.parameters(), lr=learning_scheduler_config["max_learning_rate"])
    optimizer.defaults["lr"] = get_lr_cosine_schedule(it=0, 
                                                      max_learning_rate=learning_scheduler_config['max_learning_rate'],
                                                      min_learning_rate=learning_scheduler_config['min_learning_rate'],
                                                      warmup_iters=learning_scheduler_config['warmup_iters'],
                                                      cosine_cycle_iters=learning_scheduler_config['cosine_cycle_iters'])

    epochs = 3
    train(model, device, data_loader, optimizer, 
          learning_scheduler_config=learning_scheduler_config, epochs=epochs)

    wandb.finish()

if __name__ == "__main__":
    main()
