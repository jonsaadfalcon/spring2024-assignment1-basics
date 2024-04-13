
from tests.adapters import run_train_bpe
import json
import os

dataset = "TinyStoriesV2-GPT4-train"
#dataset = "owt_train"

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

# Convert bytes in tuples to base64 strings for JSON serialization
encoded_tuples = [(base64.b64encode(item[0]).decode('utf-8'), base64.b64encode(item[1]).decode('utf-8')) for item in merges_overall]

with open(f"data/{dataset}/merges_overall.json", 'w', encoding='utf-8') as f:
    json.dump(encoded_tuples, f, indent=4)

#######################################################

longest_token = max(vocabulary.values(), key=lambda x: len(x) if hasattr(x, '__len__') else 0)
print("The longest value in the dictionary is:", longest_token)