import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel #, AdamW
import wandb
from tqdm import tqdm
import os

from tests.tokenizer import Tokenizer
from tests.Transformer import Transformer_LM
from tests.optimizer import AdamW, gradient_clipping, get_lr_cosine_schedule

##########################################################

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128, max_training_examples=10):
        
        self.examples = []

        print("Loading training text!")
        with open(file_path, 'rb') as f:
    
            for line in tqdm(f.readlines()[10:max_training_examples + 10]):
                
                #print("Line:", line.decode('utf-8'))
                line_in_bytes = line
                line_decoded = line_in_bytes.decode('utf-8')
                tokens = tokenizer.encode(line_decoded)

                #breakpoint()

                for i in range(0, max(block_size, len(tokens) - block_size + 1), block_size):
                    tensor_training_example = torch.tensor(tokens[i:i + block_size], dtype=torch.long)#[:10]
                    if len(tensor_training_example) < block_size:
                        continue
                    else:
                        self.examples.append(tensor_training_example)
                    #breakpoint()

        ##########################################################

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def train(model, device, loader, optimizer, learning_scheduler_config, model_config, saving_interval=1000, epochs=3, logging_interval=10):
    model.train()
    overall_training_count = 0
    for epoch in range(epochs):  # run for more epochs depending on dataset size
        for idx, input_ids in enumerate(loader):

            overall_training_count += 1
            
            input_ids = input_ids.to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            gradient_clipping(model.parameters(), max_l2_norm=1.0)

            #breakpoint()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            optimizer.defaults["lr"] = get_lr_cosine_schedule(it=overall_training_count, 
                                                              max_learning_rate=learning_scheduler_config['max_learning_rate'],
                                                              min_learning_rate=learning_scheduler_config['min_learning_rate'],
                                                              warmup_iters=learning_scheduler_config['warmup_iters'],
                                                              cosine_cycle_iters=learning_scheduler_config['cosine_cycle_iters'])
            
            #print("optimizer.defaults['lr']: ", optimizer.defaults['lr'])

            if idx % logging_interval == 0:
                wandb.log({"loss": loss.item()})
                print(f"Epoch: {epoch}, Idx: {idx}, Loss: {loss.item()}, LR: {optimizer.defaults['lr']}")

            if overall_training_count % saving_interval == 0:
                Transformer_LM.save_checkpoint(model=model,
                                               optimizer=optimizer,
                                               iteration=overall_training_count,
                                               out=model_config["save_path"] + f"checkpoint_{overall_training_count}.pt")

    return overall_training_count

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
        "save_path": "transformers_saved/testing/"
    }

    learning_scheduler_config = {
        "max_learning_rate": 5e-5,
        "min_learning_rate": 1e-5,
        "warmup_iters": 1000,
        "cosine_cycle_iters": 10000
    }

    epochs = 1
    saving_interval = 1000
    max_training_examples = 10

    ##################################################

    tokenizer = Tokenizer.from_files(vocab_filepath='tokenizer_saved/ts_vocab.txt',
                                     merges_filepath='tokenizer_saved/ts_merges.txt', 
                                     special_tokens=["|endoftext|"])

    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    dataset = TextDataset(file_path, tokenizer, block_size=model_config["context_length"], max_training_examples=max_training_examples)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    ##################################################

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
    
    if not os.path.exists(model_config["save_path"]):
        os.makedirs(model_config["save_path"])
    
    ##################################################
    
    #optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer = AdamW(model.parameters(), lr=learning_scheduler_config["max_learning_rate"])
    optimizer.defaults["lr"] = get_lr_cosine_schedule(it=0, 
                                                      max_learning_rate=learning_scheduler_config['max_learning_rate'],
                                                      min_learning_rate=learning_scheduler_config['min_learning_rate'],
                                                      warmup_iters=learning_scheduler_config['warmup_iters'],
                                                      cosine_cycle_iters=learning_scheduler_config['cosine_cycle_iters'])

    final_iteration = train(model, device, data_loader, optimizer, 
                            learning_scheduler_config=learning_scheduler_config, 
                            model_config=model_config, saving_interval=saving_interval,
                            epochs=epochs)
    
    print("Saving model!")
    Transformer_LM.save_checkpoint(model=model,
                                   optimizer=optimizer,
                                   iteration=final_iteration,
                                   out=model_config["save_path"] + f"checkpoint_{final_iteration}.pt")
    print("Saved model to: ", model_config["save_path"])

    breakpoint()

    logits_for_prediction = model(model.examples[0].unsqueeze(0).to(device))
    model.decode_text_from_logits(logits=logits_for_prediction,
                                  tokenizer=tokenizer,
                                  max_length=100,
                                  end_of_text_token_id=tokenizer.token_to_id("|endoftext|"))

    wandb.finish()

if __name__ == "__main__":
    main()
