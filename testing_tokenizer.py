
from tests.adapters import run_train_bpe

breakpoint()
run_train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])