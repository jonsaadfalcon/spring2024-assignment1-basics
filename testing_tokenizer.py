
from tests.adapters import run_train_bpe
import json
import os

dataset = "TinyStoriesV2-GPT4-valid" #"owt_train"

if not os.path.exists(f"data/{dataset}"):
    os.makedirs(f"data/{dataset}")

vocabulary, merges_overall = run_train_bpe(f"data/{dataset}.txt", 10000, ["<|endoftext|>"])
    
#######################################################

# Save vocabulary to json file
#with open(f"data/{dataset}vocabulary.json", "w") as f:
#    json.dump(vocabulary, f)

import base64
int_to_bytes_dict = {}
for key, value in vocabulary.items():
    # Convert bytes to base64 encoded string
    int_to_bytes_dict[key] = base64.b64encode(value).decode('utf-8')

# Save the dictionary to a JSON file
with open(f"data/{dataset}vocabulary.json", 'w', encoding='utf-8') as file:
    json.dump(int_to_bytes_dict, file, indent=4)

#######################################################

# Save merges_overall to json file
#with open(f"data/{dataset}/merges_overall.json", "w") as f:
#    json.dump(merges_overall, f)

#breakpoint()

with open(f"data/{dataset}/merges_overall.json", 'wb') as file:
    # Write each byte sequence to the file
    for byte_sequence in merges_overall:
        file.write(byte_sequence + b'\n')