
from tests.adapters import run_train_bpe
import json
import os

dataset = "TinyStoriesV2-GPT4-valid" #"owt_train"

if not os.path.exists(f"data/{dataset}"):
    os.makedirs(f"data/{dataset}")

vocabulary, merges_overall = run_train_bpe(f"data/{dataset}.txt", 10000, ["<|endoftext|>"])
    
breakpoint()

# Save vocabulary to json file
with open(f"data/{dataset}vocabulary.json", "w") as f:
    json.dump(vocabulary, f)

# Save merges_overall to json file
with open(f"data/{dataset}/merges_overall.json", "w") as f:
    json.dump(merges_overall, f)
